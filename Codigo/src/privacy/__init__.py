# -*- coding: utf-8 -*-
"""Módulo de privacidad y cumplimiento Ley 1581 de Colombia."""

from .guard import PrivacyGuard, PrivacyViolationError

__all__ = ["PrivacyGuard", "PrivacyViolationError"]
