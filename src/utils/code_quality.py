#!/usr/bin/env python3
import os
import sys
import time
import threading
import re
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (  # noqa: E402
    print_colored, run_command, run_progress_spinner,
    SPINNER_CHARS,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def start_spinner(message="Analyzing files with flake8..."):
    """
    Start a spinner animation in a separate thread.
    
    Args:
        message: Text message to display alongside the spinner
    
    Returns:
        tuple: (stop_event, spinner_thread) - use the event to stop the spinner
    """
    stop_event = threading.Event()
    
    spinner_thread = threading.Thread(
        target=run_progress_spinner,
        args=(message, stop_event)
    )
    spinner_thread.daemon = True
    spinner_thread.start()
    
    return stop_event, spinner_thread


def stop_spinner(stop_event, spinner_thread):
    """
    Stop a running spinner animation.
    
    Args:
        stop_event: Event to signal stopping the spinner
        spinner_thread: The thread running the spinner
    """
    stop_event.set()
    spinner_thread.join(0.2)
    print("\r" + " " * 50 + "\r", end="")  # Clear the spinner line


def parse_flake8_output(output, project_dir, file_issues, file_issue_types,
                        issue_type_counts, filter_code=None):
    """
    Parse flake8 output and update issue tracking data structures.
    
    Args:
        output: The stdout from flake8
        project_dir: The project directory path
        file_issues: Dictionary of files to total issue counts
        file_issue_types: Nested dictionary of files to issue types to counts
        issue_type_counts: Dictionary of issue types to total counts
        filter_code: Only count issues with this code (e.g., 'E501')
        
    Returns:
        int: Number of issues found and processed
    """
    issues_found = 0
    
    if output and ":" in output:
        lines = output.strip().split('\n')
        for line in lines:
            if ":" in line:
                # Pattern to extract file path, line number, and issue code
                match = re.match(r'^(.*?):(\d+):(\d+): ([A-Z]\d+) (.*)$', line)
                if match:
                    file_path, row, col, issue_code, issue_text = match.groups()
                    
                    # Skip if we're filtering for a specific code
                    if filter_code and issue_code != filter_code:
                        continue
                    
                    # Get the relative path or basename
                    file_name = os.path.relpath(file_path, project_dir)
                    
                    # Initialize file in tracking dict if not present
                    if file_name not in file_issues:
                        file_issues[file_name] = 0
                    
                    # Increment issue counts
                    file_issues[file_name] += 1
                    file_issue_types[file_name][issue_code] += 1
                    issue_type_counts[issue_code] += 1
                    issues_found += 1
    
    return issues_found


def run_flake8(config, project_dir, verbose=False):
    """
    Run flake8 to check code quality and show a detailed summary
    with files and their issue counts broken down by error type.

    Args:
        config: Dictionary containing configuration parameters
        project_dir: Path to the project directory to analyze
        verbose: If True, show debugging info and raw outputs

    Returns:
        bool: True if the analysis completed successfully
    """
    print_colored('\n🔍 Running flake8 code quality check', BLUE)

    # Validate config
    required_keys = ['use_conda', 'pip_bin']
    if config['use_conda']:
        required_keys.append('env_path')
    
    for key in required_keys:
        if key not in config:
            print_colored(f"Error: Missing required config key '{key}'", RED)
            return False

    # Install flake8 if not already installed
    if config['use_conda']:
        pip_bin = os.path.join(config['env_path'], 'bin', 'pip')
        flake8_bin = os.path.join(config['env_path'], 'bin', 'flake8')
    else:
        pip_bin = config['pip_bin']
        flake8_bin = os.path.join(os.path.dirname(config['pip_bin']), 'flake8')

    # Check if flake8 is installed
    if not os.path.exists(flake8_bin):
        print_colored("Installing flake8...", YELLOW)
        run_command(f"{pip_bin} install flake8", shell=True)

    # Get list of Python files in the project
    print_colored(f"Scanning Python files in {project_dir}...", GREEN)
    stop_event, spinner = start_spinner()
    
    find_py_files_cmd = f"find {project_dir} -name '*.py'"
    py_files_result = run_command(
        find_py_files_cmd, shell=True, capture_output=True
    )

    if py_files_result and py_files_result.stdout:
        python_files = py_files_result.stdout.strip().split('\n')
        files_count = len(python_files)
    else:
        python_files = []
        files_count = 0

    # Count total lines in all Python files
    lines_count_cmd = (
        f"find {project_dir} -name '*.py' -exec cat {{}} \\; | wc -l"
    )
    lines_count_result = run_command(
        lines_count_cmd, shell=True, capture_output=True
    )

    if lines_count_result and lines_count_result.stdout:
        lines_count = int(lines_count_result.stdout.strip())
    else:
        lines_count = 0
    
    stop_spinner(stop_event, spinner)

    # Data structures to store issue statistics
    file_issues = {}  # Total issues per file
    file_issue_types = defaultdict(lambda: defaultdict(int))  # Issues per file
    issue_type_counts = defaultdict(int)  # Total counts per issue type
    total_issues = 0

    # Run standard flake8 check first (excluding E501)
    print_colored("\nRunning flake8 code style check...", BLUE)
    stop_event, spinner = start_spinner()
    
    format_str = "%(path)s:%(row)d:%(col)d: %(code)s %(text)s"
    
    # Use double quotes in the command to avoid issues with single quotes
    detailed_cmd = (
        f"{flake8_bin} {project_dir} --count --exit-zero "
        f"--max-complexity=10 --ignore=E501 "
        f'--statistics --format="{format_str}"'
    )

    if verbose:
        print_colored(f"Running command: {detailed_cmd}", CYAN)

    detailed_result = run_command(
        detailed_cmd, shell=True, capture_output=True
    )
    
    stop_spinner(stop_event, spinner)

    if detailed_result and detailed_result.stdout:
        if verbose:
            print_colored("\nRaw flake8 output:", CYAN)
            print(detailed_result.stdout)
        
        # Parse general flake8 output
        issues = parse_flake8_output(
            detailed_result.stdout, 
            project_dir, 
            file_issues, 
            file_issue_types, 
            issue_type_counts
        )
        total_issues += issues
    
    # Now run a separate check for E501 (line too long) errors
    print_colored("\nChecking for line length (E501) issues...", BLUE)
    stop_event, spinner = start_spinner()
    
    # Run flake8 with only E501 enabled - using double quotes for format
    e501_cmd = (
        f"{flake8_bin} {project_dir} --count --exit-zero "
        f"--select=E501 --max-line-length=79 "
        f'--format="{format_str}"'
    )
    
    if verbose:
        print_colored(f"Running E501 command: {e501_cmd}", CYAN)
    
    e501_result = run_command(
        e501_cmd, shell=True, capture_output=True
    )
    
    stop_spinner(stop_event, spinner)
    
    if e501_result and e501_result.stdout:
        if verbose:
            print_colored("\nRaw E501 output:", CYAN)
            print(e501_result.stdout)
        
        # Parse E501 flake8 output
        e501_issues = parse_flake8_output(
            e501_result.stdout, 
            project_dir, 
            file_issues, 
            file_issue_types, 
            issue_type_counts,
            filter_code="E501"
        )
        total_issues += e501_issues
        
        if verbose:
            print_colored(f"Found {e501_issues} E501 issues", CYAN)

    # Print results
    if total_issues > 0:
        # Summary at the end
        print_colored("\n📊 Analysis Summary:", BLUE)
        print_colored(f"   - Files analyzed: {files_count}", GREEN)
        print_colored(f"   - Lines analyzed: {lines_count}", GREEN)
        
        # Determine if there are critical issues (errors vs warnings)
        has_critical_issues = any(
            count > 0 for code, count in issue_type_counts.items() 
            if code.startswith(('E', 'F')) and code != 'E501'  # E501 not critical
        )
        
        error_color = RED if has_critical_issues else GREEN
        print_colored(
            f"   - Critical errors: {'YES' if has_critical_issues else 'No'}",
            error_color
        )

        issue_color = YELLOW if total_issues > 0 else GREEN
        print_colored(f"   - Style issues: {total_issues}", issue_color)

        # Display issues by type
        print_colored("\n📋 Issues by type:", BLUE)
        sorted_issues = sorted(
            issue_type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for issue_code, count in sorted_issues:
            color = RED if issue_code.startswith(('E', 'F')) and issue_code != 'E501' else YELLOW
            code_description = get_error_description(issue_code)
            print_colored(f"   - {issue_code} ({code_description}): {count}", color)

        # Display issues per file
        print_colored("\n📋 Issues per file:", BLUE)
        
        # Create a dictionary with all Python files found and set issues to 0 if not in file_issues
        all_files = {}
        for file_path in python_files:
            file_name = os.path.relpath(file_path, project_dir)
            if file_name in file_issues:
                all_files[file_name] = file_issues[file_name]
            else:
                all_files[file_name] = 0
        
        # Sort by issue count, with files that have issues first
        sorted_files = sorted(
            all_files.items(),
            key=lambda x: (x[1] == 0, -x[1])  # First non-zero issues (descending), then zero issues
        )

        for file_name, issue_count in sorted_files:
            color = RED if issue_count >= 1 else GREEN
            print_colored(f"   - {file_name}: {issue_count} issues", color)
            
            # Show breakdown of issue types for each file (only for files with issues)
            if issue_count > 0:
                file_issues_sorted = sorted(
                    file_issue_types[file_name].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                # Show all issues for each file
                for issue_code, count in file_issues_sorted:
                    code_description = get_error_description(issue_code)
                    print_colored(f"     - {issue_code} ({code_description}): {count}", CYAN)
    else:
        print_colored("✅ No style issues found", GREEN)

    return True


def get_error_description(code):
    """
    Get a description for a flake8 error code.
    
    Args:
        code: The error code (e.g. 'E501')
        
    Returns:
        str: Description of the error code
    """
    error_descriptions = {
        # E errors (PEP8)
        'E101': 'indentation contains mixed spaces and tabs',
        'E111': 'indentation is not a multiple of 4',
        'E112': 'expected an indented block',
        'E113': 'unexpected indentation',
        'E114': 'indentation is not a multiple of 4 (comment)',
        'E115': 'expected an indented block (comment)',
        'E116': 'unexpected indentation (comment)',
        'E121': 'continuation line under-indented for hanging indent',
        'E122': 'continuation line missing indentation or outdented',
        'E123': 'closing bracket does not match indentation of opening bracket',
        'E124': 'closing bracket does not match visual indentation',
        'E125': 'continuation line with same indent as next logical line',
        'E126': 'continuation line over-indented for hanging indent',
        'E127': 'continuation line over-indented for visual indent',
        'E128': 'continuation line under-indented for visual indent',
        'E129': 'visually indented line with same indent as next logical line',
        'E131': 'continuation line unaligned for hanging indent',
        'E133': 'closing bracket is missing indentation',
        'E201': 'whitespace after (',
        'E202': 'whitespace before )',
        'E203': 'whitespace before :',
        'E211': 'whitespace before (',
        'E221': 'multiple spaces before operator',
        'E222': 'multiple spaces after operator',
        'E223': 'tab before operator',
        'E224': 'tab after operator',
        'E225': 'missing whitespace around operator',
        'E226': 'missing whitespace around arithmetic operator',
        'E227': 'missing whitespace around bitwise or shift operator',
        'E228': 'missing whitespace around modulo operator',
        'E231': 'missing whitespace after ,',
        'E241': 'multiple spaces after ,',
        'E242': 'tab after ,',
        'E251': 'unexpected spaces around keyword / parameter equals',
        'E261': 'at least two spaces before inline comment',
        'E262': 'inline comment should start with # ',
        'E265': 'block comment should start with # ',
        'E266': 'too many leading # for block comment',
        'E271': 'multiple spaces after keyword',
        'E272': 'multiple spaces before keyword',
        'E273': 'tab after keyword',
        'E274': 'tab before keyword',
        'E275': 'missing whitespace after keyword',
        'E301': 'expected 1 blank line, found 0',
        'E302': 'expected 2 blank lines, found 0',
        'E303': 'too many blank lines',
        'E304': 'blank lines found after function decorator',
        'E305': 'expected 2 blank lines after class or function definition',
        'E306': 'expected 1 blank line before a nested definition',
        'E401': 'multiple imports on one line',
        'E402': 'module level import not at top of file',
        'E501': 'line too long',
        'E502': 'the backslash is redundant between brackets',
        'E701': 'multiple statements on one line (colon)',
        'E702': 'multiple statements on one line (semicolon)',
        'E703': 'statement ends with a semicolon',
        'E704': 'multiple statements on one line (def)',
        'E711': 'comparison to None should be if cond is None:',
        'E712': 'comparison to True should be if cond is True:',
        'E713': 'test for membership should be not in',
        'E714': 'test for object identity should be is not',
        'E721': 'do not compare types, use isinstance()',
        'E722': 'do not use bare except, specify exception instead',
        'E731': 'do not assign a lambda expression, use a def',
        'E741': 'do not use variables named l, O, or I',
        'E742': 'do not define classes named l, O, or I',
        'E743': 'do not define functions named l, O, or I',
        'E901': 'SyntaxError or IndentationError',
        'E902': 'IOError',
        
        # F errors (PyFlakes)
        'F401': 'module imported but unused',
        'F402': 'import module from line N shadowed by loop variable',
        'F403': 'from module import * used; unable to detect undefined names',
        'F404': 'future import(s) name after other statements',
        'F405': 'name may be undefined, or defined from star imports',
        'F406': 'from module import * only allowed at module level',
        'F407': 'an undefined __future__ feature name was imported',
        'F541': 'f-string is missing placeholders',
        'F601': 'dictionary key name repeated with different values',
        'F602': 'dictionary key variable name repeated with different values',
        'F621': 'too many expressions in an assignment with star-unpacking',
        'F622': 'two or more starred expressions in an assignment',
        'F631': 'assertion test is a tuple, which is always True',
        'F632': 'use ==/!= to compare str, bytes, and int literals',
        'F633': 'use of >> is invalid with print function',
        'F634': 'if test is a tuple, which is always True',
        'F701': 'a break statement outside of a while or for loop',
        'F702': 'a continue statement outside of a while or for loop',
        'F703': 'a continue statement in a finally block in a loop',
        'F704': 'a yield or yield from statement outside of a function',
        'F705': 'a return statement outside of a function/method',
        'F706': 'a return statement in a generator function that returns a value',
        'F707': 'an except: block as not the last exception handler',
        'F811': 'redefinition of unused name from line N',
        'F812': 'list comprehension redefines name from line N',
        'F821': 'undefined name',
        'F822': 'undefined name in __all__',
        'F823': 'local variable name referenced before assignment',
        'F831': 'duplicate argument name in function definition',
        'F841': 'local variable is assigned to but never used',
        
        # W warnings
        'W191': 'indentation contains tabs',
        'W291': 'trailing whitespace',
        'W292': 'no newline at end of file',
        'W293': 'blank line contains whitespace',
        'W391': 'blank line at end of file',
        'W503': 'line break before binary operator',
        'W504': 'line break after binary operator',
        'W505': 'doc line too long',
        'W601': '.has_key() is deprecated, use in',
        'W602': 'deprecated form of raising exception',
        'W603': '<> is deprecated, use !=',
        'W604': 'backticks are deprecated, use repr()',
        'W605': 'invalid escape sequence',
        'W606': 'async and await are reserved keywords',
        
        # C complexity
        'C901': 'function/method is too complex',
    }
    
    return error_descriptions.get(code, 'unknown error')


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    print_colored(
        "This script is intended to be imported, not run directly.",
        YELLOW
    )
    print_colored(
        "For testing, please import this module in another script.",
        YELLOW
    )
