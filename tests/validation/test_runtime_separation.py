"""Independent serial reference code stays separate from production transfer."""

import ast
from pathlib import Path

import prom3theus.inversion as inversion
import prom3theus.inversion.assembly as assembly
import prom3theus.rt as rt


def test_production_public_apis_expose_parallel_transfer_without_serial_reference():
    assert rt.LTELocalCoefficients is not None
    assert callable(rt.formal_stokes)
    for name in ("LTESynthesizer", "PolarizedFormalSolver", "RayDistancePath", "TransferPath"):
        assert not hasattr(rt, name)
    for name in ("LTEForwardComposition", "LTESynthesisBackend", "ForwardSynthesisBackend", "ForwardRuntime"):
        assert not hasattr(inversion, name)
    assert not hasattr(assembly, "ForwardAssembly")
    assert not hasattr(assembly, "build_forward_assembly")


def test_production_training_and_observation_modules_do_not_import_numerical_references():
    package = Path(rt.__file__).resolve().parents[0]
    # Application and diagnostic entry points may explicitly launch validation
    # jobs; the scene, observation term and training execution may not.
    package = package.parent
    for directory in ("rt", "inversion", "training", "components"):
        for path in (package / directory).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                assert not any("prom3theus.validation" in name for name in modules), path
