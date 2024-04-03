# File Layout

Most Python projects have a more or less standardized structure, e.g.

- [Python Application Layouts: A Reference](https://realpython.com/python-application-layouts/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

We roughly follow the Cookiecutter template, with `src` renamed to `dl`, some files missing and the following directories

- `dl`: All reusable code goes here. In future assignments, you can add to the files already present or create new ones.
- `assignments`: Put your assignments here. If you number them with {assignment number}_{problem number}_assignment_description.py, you file manager will order them correctly.
- `data`: Downloaded data (later).
- `models`: Saved Models (later).
- `requirements.txt`: Required packages. Can be used to install them automatically with `pip` (you likely have them already).


# Python Setup

The included example in `assignments` may not run because it cannot find the parts split off into the directory `dl`. Below are some instructions for different environments.

## Shell

From the base directory you can run as a module

    python -m assignments.1_1_directory_setup

or set an environment variable that lets Python know where to find `dl`

    export PYTHONPATH=path_to_project_folder/
    python assignments/1_1_directory_setup.py

## VS Code

Add the file `.env` to the project root directory with content

    PYTHONPATH=path_to_project_folder/

and `.vscode/settings.json` with content

    {
        "terminal.integrated.env.linux": {
            "PYTHONPATH": "${workspaceFolder}"
        }
    }%

Adapt the paths and enventually change `linux` to `osx` or `windows`.

## Spyder

You can set the `PYTHONPATH` in the menu `Tools -> PYTHONPATH Manager` (at least in my version, which is a little older).

# Version Control

I recommend to use a version control system like Git or Github for your files. They allow you to restore previous versions of your project and avoid files like `dl.models_old4.py`, etc.. They are not tied to Python, so you can use them with Latex for your thesis, too.

