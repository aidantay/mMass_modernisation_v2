# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#     Copyright (C) 2005-2013 Martin Strohalm <www.mmass.org>
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     Complete text of GNU GPL can be found in the file LICENSE.TXT in the
#     main directory of the program.
# -------------------------------------------------------------------------

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import mspy.mod_stopper


# ============================================================================
# Module-level and function-level fixtures for state isolation
# ============================================================================

@pytest.fixture(scope="module", autouse=True)
def reset_stopper_module_start():
    """Module-scoped autouse fixture to reset stopper at module start and end."""
    mspy.mod_stopper.start()
    yield
    mspy.mod_stopper.start()


@pytest.fixture(scope="function", autouse=True)
def ensure_stopper_clean():
    """Function-scoped autouse fixture to reset stopper before each test."""
    mspy.mod_stopper.start()
    yield


# ============================================================================
# Smoke test: verify all symbols exist
# ============================================================================

def test_import_mod_stopper():
    """Smoke test: verify module can be imported and all symbols exist."""
    assert hasattr(mspy.mod_stopper, 'ForceQuit')
    assert hasattr(mspy.mod_stopper, 'stopper')
    assert hasattr(mspy.mod_stopper, 'STOPPER')
    assert hasattr(mspy.mod_stopper, 'CHECK_FORCE_QUIT')
    assert hasattr(mspy.mod_stopper, 'stop')
    assert hasattr(mspy.mod_stopper, 'start')


# ============================================================================
# ForceQuit exception tests
# ============================================================================

def test_forcequit_is_exception():
    """Test that ForceQuit is a subclass of Exception."""
    assert issubclass(mspy.mod_stopper.ForceQuit, Exception)


def test_forcequit_raise_and_catch():
    """Test that ForceQuit can be raised and caught."""
    with pytest.raises(mspy.mod_stopper.ForceQuit):
        raise mspy.mod_stopper.ForceQuit()


def test_forcequit_raise_with_message():
    """Test that ForceQuit can be raised with a message argument."""
    with pytest.raises(mspy.mod_stopper.ForceQuit) as exc_info:
        raise mspy.mod_stopper.ForceQuit("Test message")
    assert str(exc_info.value) == "Test message"


# ============================================================================
# stopper.__init__ tests
# ============================================================================

def test_stopper_init_creates_instance():
    """Test that stopper() creates a stopper instance."""
    s = mspy.mod_stopper.stopper()
    assert isinstance(s, mspy.mod_stopper.stopper)


def test_stopper_init_value_is_false():
    """Test that stopper.__init__ sets value to False."""
    s = mspy.mod_stopper.stopper()
    assert s.value is False


# ============================================================================
# stopper.__nonzero__ (Python 2 boolean protocol) tests
# ============================================================================

def test_stopper_nonzero_false_initially():
    """Test that bool(stopper) is False initially."""
    s = mspy.mod_stopper.stopper()
    assert not s
    assert bool(s) is False


def test_stopper_nonzero_true_after_enable():
    """Test that bool(stopper) is True after enable()."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    assert s
    assert bool(s) is True


def test_stopper_nonzero_false_after_disable():
    """Test that bool(stopper) is False after disable()."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.disable()
    assert not s
    assert bool(s) is False


# ============================================================================
# stopper.__repr__ tests
# ============================================================================

def test_stopper_repr_false_initially():
    """Test that repr(stopper) is 'False' initially."""
    s = mspy.mod_stopper.stopper()
    assert repr(s) == 'False'


def test_stopper_repr_true_after_enable():
    """Test that repr(stopper) is 'True' after enable()."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    assert repr(s) == 'True'


def test_stopper_repr_false_after_disable():
    """Test that repr(stopper) is 'False' after disable()."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.disable()
    assert repr(s) == 'False'


# ============================================================================
# stopper.enable() tests
# ============================================================================

def test_stopper_enable_sets_value_true():
    """Test that enable() sets value to True."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    assert s.value is True


def test_stopper_enable_idempotent():
    """Test that calling enable() twice has the same effect as once."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.enable()
    assert s.value is True


# ============================================================================
# stopper.disable() tests
# ============================================================================

def test_stopper_disable_sets_value_false():
    """Test that disable() sets value to False."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.disable()
    assert s.value is False


def test_stopper_disable_idempotent():
    """Test that calling disable() twice has the same effect as once."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.disable()
    s.disable()
    assert s.value is False


# ============================================================================
# stopper.check() - False branch tests
# ============================================================================

def test_stopper_check_false_does_not_raise():
    """Test that check() does not raise when value is False."""
    s = mspy.mod_stopper.stopper()
    s.check()  # Should not raise


def test_stopper_check_false_value_stays_false():
    """Test that check() keeps value as False when initially False."""
    s = mspy.mod_stopper.stopper()
    s.check()
    assert s.value is False


# ============================================================================
# stopper.check() - True branch tests
# ============================================================================

def test_stopper_check_true_raises_forcequit():
    """Test that check() raises ForceQuit when value is True."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    with pytest.raises(mspy.mod_stopper.ForceQuit):
        s.check()


def test_stopper_check_true_resets_to_false():
    """Test that check() resets value to False after raising ForceQuit."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    try:
        s.check()
    except mspy.mod_stopper.ForceQuit:
        pass
    assert s.value is False


def test_stopper_check_second_call_does_not_raise():
    """Test that second call to check() does not raise after reset."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    try:
        s.check()
    except mspy.mod_stopper.ForceQuit:
        pass
    s.check()  # Should not raise


# ============================================================================
# STOPPER singleton tests
# ============================================================================

def test_stopper_singleton_is_instance():
    """Test that STOPPER is an instance of stopper class."""
    assert isinstance(mspy.mod_stopper.STOPPER, mspy.mod_stopper.stopper)


def test_stopper_singleton_initial_state():
    """Test that STOPPER initial state is False after start()."""
    mspy.mod_stopper.start()
    assert mspy.mod_stopper.STOPPER.value is False


def test_check_force_quit_is_stopper_check():
    """Test that CHECK_FORCE_QUIT is a reference to STOPPER.check."""
    assert mspy.mod_stopper.CHECK_FORCE_QUIT == mspy.mod_stopper.STOPPER.check


# ============================================================================
# stop()/start() module-level function tests
# ============================================================================

def test_stop_sets_stopper_value_true():
    """Test that stop() sets STOPPER.value to True."""
    mspy.mod_stopper.stop()
    assert mspy.mod_stopper.STOPPER.value is True


def test_start_sets_stopper_value_false():
    """Test that start() sets STOPPER.value to False."""
    mspy.mod_stopper.stop()
    mspy.mod_stopper.start()
    assert mspy.mod_stopper.STOPPER.value is False


def test_stop_then_check_force_quit_raises():
    """Test that stop() followed by CHECK_FORCE_QUIT() raises ForceQuit."""
    mspy.mod_stopper.stop()
    with pytest.raises(mspy.mod_stopper.ForceQuit):
        mspy.mod_stopper.CHECK_FORCE_QUIT()


def test_stop_then_check_force_quit_resets():
    """Test that CHECK_FORCE_QUIT() resets STOPPER after raising."""
    mspy.mod_stopper.stop()
    try:
        mspy.mod_stopper.CHECK_FORCE_QUIT()
    except mspy.mod_stopper.ForceQuit:
        pass
    assert mspy.mod_stopper.STOPPER.value is False


def test_start_then_check_force_quit_does_not_raise():
    """Test that start() followed by CHECK_FORCE_QUIT() does not raise."""
    mspy.mod_stopper.start()
    mspy.mod_stopper.CHECK_FORCE_QUIT()  # Should not raise


# ============================================================================
# State transition tests
# ============================================================================

def test_stopper_enable_disable_enable_cycle():
    """Test enable -> disable -> enable cycle maintains correct state."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    assert s.value is True
    s.disable()
    assert s.value is False
    s.enable()
    assert s.value is True


def test_stopper_check_after_multiple_enables():
    """Test check() after multiple consecutive enables."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    s.enable()
    s.enable()
    with pytest.raises(mspy.mod_stopper.ForceQuit):
        s.check()
    assert s.value is False


def test_stopper_disable_after_check_reset():
    """Test disable() after check() has reset the value."""
    s = mspy.mod_stopper.stopper()
    s.enable()
    try:
        s.check()
    except mspy.mod_stopper.ForceQuit:
        pass
    s.disable()
    assert s.value is False


# ============================================================================
# Hypothesis property-based tests
# ============================================================================

@given(st.booleans())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_stopper_enable_disable_produces_valid_bool(enabled):
    """Test that enable/disable sequences always produce valid bool values."""
    s = mspy.mod_stopper.stopper()
    if enabled:
        s.enable()
    else:
        s.disable()
    # Verify result is always a boolean
    result = bool(s)
    assert isinstance(result, bool)
    assert result is (s.value is True)


@given(st.booleans())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_stopper_check_always_resets_value_to_false(enabled):
    """Test that after check(), value is always False (regardless of initial state)."""
    s = mspy.mod_stopper.stopper()
    if enabled:
        s.enable()

    try:
        s.check()
    except mspy.mod_stopper.ForceQuit:
        pass

    # After check(), value must always be False
    assert s.value is False


@given(st.lists(st.booleans(), min_size=1, max_size=10))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_stopper_state_transitions_hypothesis(operations):
    """Test stopper with random sequences of enable/disable operations."""
    s = mspy.mod_stopper.stopper()
    for op in operations:
        if op:
            s.enable()
        else:
            s.disable()

    # After operations, value should match the last operation
    expected_value = operations[-1] if operations else False
    assert s.value is expected_value
    assert bool(s) is expected_value


# ============================================================================
# Integration tests with singleton STOPPER
# ============================================================================

def test_singleton_stopper_and_stop_start_integration():
    """Test integration of singleton STOPPER with stop() and start()."""
    mspy.mod_stopper.start()
    assert mspy.mod_stopper.STOPPER.value is False

    mspy.mod_stopper.stop()
    assert mspy.mod_stopper.STOPPER.value is True

    mspy.mod_stopper.start()
    assert mspy.mod_stopper.STOPPER.value is False


def test_check_force_quit_function_references_singleton():
    """Test that CHECK_FORCE_QUIT function operates on singleton STOPPER."""
    mspy.mod_stopper.stop()
    assert mspy.mod_stopper.STOPPER.value is True

    try:
        mspy.mod_stopper.CHECK_FORCE_QUIT()
    except mspy.mod_stopper.ForceQuit:
        pass

    assert mspy.mod_stopper.STOPPER.value is False


def test_multiple_stopper_instances_are_independent():
    """Test that multiple stopper instances are independent."""
    s1 = mspy.mod_stopper.stopper()
    s2 = mspy.mod_stopper.stopper()

    s1.enable()
    assert s1.value is True
    assert s2.value is False

    s2.enable()
    assert s1.value is True
    assert s2.value is True

    s1.disable()
    assert s1.value is False
    assert s2.value is True
