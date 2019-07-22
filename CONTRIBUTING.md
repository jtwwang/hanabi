# Contributing

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

### Table of contents
[What should I know before I get started?](#what-should-i-know-before-i-get-started)

[How can I contribute?](#how-can-i-contribute)
* [Pull Requests](#pull-requests)
* [Suggesting Enhancements](#suggesting-enhancements)
* [Reporting Bugs](#reporting-bugs)

[Style Guide](#style-guide)
* [Git Commit Messages](#git-commit-messages)
* [Python Documentation](#python-documentation)

## What should I know before I get started?

This repository is result of research done in the attempt of solving the game of Hanabi, and in particular the ad-hoc challenge. It is intentionally opened to everyone under the [Apache 2.0 license](https://github.com/jtwwang/hanabi/blob/master/LICENSE) with the hope it can be useful to students and researchers in the field. As it is an open-source project, we are happy to accept contributions for both fixing bugs and introducing new features. However, differently from other open-source projects it is not meant to be used as an external package, but it is structured in such a way that is possible to both reproduce our results, and - if desired - to expand it to test new ideas and algorithms.

## How can I contribute?
### Reporting Bugs
* Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/jtwwang/hanabi/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/jtwwang/hanabi/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.
* Specify your OS and python version in the issue description

### Suggesting Enhancements
Enhancement suggestions are tracked as [GitHub issues](https://github.com/jtwwang/hanabi/issues). After you've determined which repository your enhancement suggestion is related to, create an issue on that repository and provide the following information:

* Use a **clear and descriptive title** for the issue to identify the suggestion.
* Provide a step-by-step description of the suggested enhancement in as many details as possible.
* Describe the current behavior and explain **which behavior you expect** to see instead and why.
* Explain **why this enhancement would be useful**.
* If applicable, list some other repository or paper that provide context to this enhancement.

### Pull Requests
If you wrote a patch that fixes bugs, a cosmetic improvement, or added a new feature, please create a full request and provide the following information. Follow the [Style Guide](#style-guide) to make sure your new code is clearly documented.

* Use a **clear and descriptive title** for the pull request
* Describe in great detail everything that the new code does, how it changes from the existing code, and the expected behavior
* If related to an issue, reference said issue
* Use at least one label to identify clearly the intent of the pull request.

## Style Guide

### Git Commit Messages
* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Documentation
* Write a brief description of the functionality of each new function, at the beginning of the function
* Write the arguments needed (if any) and their data type for each new function
* Write the return variables (if any) and their data type for each new function

### License update
If you modify one of the files that was natively provided by Deepmind in their Hanabi Learning Environment, please specify that you edited the file. Such files can be recognized by the google copyright mark at the top of the file.
