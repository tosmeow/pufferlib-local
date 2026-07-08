import os
import sys
import types

ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)

try:
    import pufferlib._C  # noqa: F401
except Exception:
    fake_c = types.ModuleType("pufferlib._C")
    fake_c.precision_bytes = 4
    fake_c.gpu = 0
    fake_c.env_name = "docs"
    fake_c.get_nccl_id = lambda: b""
    fake_c.get_utilization = lambda gpu_id=0: {}
    sys.modules["pufferlib._C"] = fake_c

project = "PufferLib Local Notes"
author = "PufferAI contributors"
release = "4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "PufferLib Local Docs"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "",
    "dark_logo": "",
}
