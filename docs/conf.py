from datetime import datetime
from pathlib import Path

DOCS_PATH = Path(__file__).parent
ROOT_PATH = DOCS_PATH.parent

version_file = ROOT_PATH.joinpath("version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()

project = "Draftfit"
copyright = f"2025-{datetime.now().year}, SpecForge"
author = "JayYun98; SpecForge contributors (upstream engine)"

version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]


myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "substitution",
]

myst_heading_anchors = 5

myst_ref_domains = ["std", "py"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": "https://github.com/JayYun98/Draftfit",
    "path_to_docs": "docs",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "JayYun98",
    "github_repo": "dspark-train-platform",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]


htmlhelp_basename = "draftfitdoc"

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "draftfit.tex",
        "Draftfit Documentation",
        author,
        "manual",
    ),
]

man_pages = [(master_doc, "draftfit", "Draftfit Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "draftfit",
        "Draftfit Documentation",
        author,
        "Draftfit",
        "Your workload. Your draft.",
        "Miscellaneous",
    ),
]

epub_title = project

epub_exclude_files = ["search.html"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

navigation_with_keys = False
