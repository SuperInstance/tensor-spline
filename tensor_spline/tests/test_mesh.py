"""Tests for mesh.py — SuperInstance mesh registration."""

import pytest


class TestMeshRegistration:
    def test_register_tensor_spline(self):
        """Test that register_tensor_spline registers capabilities."""

        class MockRegistry:
            def __init__(self):
                self.entries = {}

            def register(self, category, name, factory):
                self.entries[(category, name)] = factory

        from tensor_spline.mesh import register_tensor_spline
        registry = MockRegistry()
        register_tensor_spline(registry)

        assert ("compressors", "spline-linear") in registry.entries
        assert ("compressors", "low-rank") in registry.entries
        assert ("compressors", "inject") in registry.entries
        assert ("compressors", "measure") in registry.entries
        assert ("compressors", "recommend") in registry.entries

    def test_registered_factories_are_callable(self):
        class MockRegistry:
            def __init__(self):
                self.entries = {}
            def register(self, category, name, factory):
                self.entries[(category, name)] = factory

        from tensor_spline.mesh import register_tensor_spline
        registry = MockRegistry()
        register_tensor_spline(registry)

        # The utility entries should be callable
        assert callable(registry.entries[("compressors", "inject")])
        assert callable(registry.entries[("compressors", "measure")])
        assert callable(registry.entries[("compressors", "recommend")])
