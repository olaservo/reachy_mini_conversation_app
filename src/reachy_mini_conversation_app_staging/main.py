"""Staging shim: give the staging app name a package dir of its own.

The daemon locates a running app's UI (and JSON-RPC relay target) by reading
``site-packages/<app_name>/main.py`` and scraping the ``custom_app_url``
assignment. The staging entry point renames the app but the real package is
still ``reachy_mini_conversation_app``, so without this shim the daemon finds
no ``main.py`` and the desktop app never shows the UI.
"""

from reachy_mini_conversation_app.main import ReachyMiniConversationApp

custom_app_url = "http://0.0.0.0:7860/"

__all__ = ["ReachyMiniConversationApp"]
