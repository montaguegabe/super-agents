"""Alias package for the ``super-agents`` distribution.

Installing ``superagents`` installs ``super-agents``; the import name is
``super_agents`` either way. This module forwards to it so that
``import superagents`` also works.
"""

import sys

import super_agents

sys.modules[__name__] = super_agents
