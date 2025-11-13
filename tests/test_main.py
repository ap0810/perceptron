"""
Unit tests for the perceptron implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

import main


class TestActivationFunction:
    """Test the step activation function."""

    def test_step_positive(self) -> None:
        """Test step function with positive input."""
        assert main.step(1.0) == 1
        assert main.step(0.5) == 1
        assert main.step(100) == 1

    def test_step_zero(self) -> None:
        """Test step function with zero input."""
        assert main.step(0) == 1

    def test_step_negative(self) -> None:
        """Test step function with negative input."""
        assert main.step(-0.1) == 0
        assert main.step(-1.0) == 0
        assert main.step(-100) == 0


class TestPredictionFunction:
    """Test the predict function."""

    def test_predict_basic(self) -> None:
        """Test basic prediction."""
        # w1=1, w2=1, bias=0: predicts 1 when x1+x2 >= 0
        result = main.predict(1, 1, 1, 1, 0)
        assert result == 1

    def test_predict_with_bias(self) -> None:
        """Test prediction with bias."""
        # Bias shifts the decision boundary
        result = main.predict(0, 0, 1, 1, -1)
        assert result == 0

    def test_predict_zero_weights(self) -> None:
        """Test prediction with zero weights."""
        result = main.predict(5, 5, 0, 0, 1)
        assert result == 1

    def test_predict_negative_weights(self) -> None:
        """Test prediction with negative weights."""
        result = main.predict(1, 1, -1, -1, 0)
        assert result == 0


class TestLogicGates:
    """Test that logic gate definitions are valid."""

    def test_logic_gates_exist(self) -> None:
        """Test that all expected logic gates are defined."""
        expected_gates = ["AND", "OR", "NAND", "NOR", "XOR"]
        for gate in expected_gates:
            assert gate in main.LOGIC_GATES

    def test_logic_gate_structure(self) -> None:
        """Test that logic gates have correct structure."""
        for gate_name, (x1, x2, y) in main.LOGIC_GATES.items():
            assert len(x1) == 4, f"{gate_name} x1 should have 4 elements"
            assert len(x2) == 4, f"{gate_name} x2 should have 4 elements"
            assert len(y) == 4, f"{gate_name} y should have 4 elements"
            assert all(val in [0, 1] for val in x1)
            assert all(val in [0, 1] for val in x2)
            assert all(val in [0, 1] for val in y)

    def test_and_gate(self) -> None:
        """Test AND gate data."""
        x1, x2, y = main.LOGIC_GATES["AND"]
        # AND: only outputs 1 when both inputs are 1
        assert y == [0, 0, 0, 1]

    def test_or_gate(self) -> None:
        """Test OR gate data."""
        x1, x2, y = main.LOGIC_GATES["OR"]
        # OR: outputs 0 only when both inputs are 0
        assert y == [0, 1, 1, 1]

    def test_xor_gate(self) -> None:
        """Test XOR gate data."""
        x1, x2, y = main.LOGIC_GATES["XOR"]
        # XOR: outputs 1 when inputs differ
        assert y == [0, 1, 1, 0]

    def test_nand_gate(self) -> None:
        """Test NAND gate data."""
        x1, x2, y = main.LOGIC_GATES["NAND"]
        # NAND: opposite of AND
        assert y == [1, 1, 1, 0]

    def test_nor_gate(self) -> None:
        """Test NOR gate data."""
        x1, x2, y = main.LOGIC_GATES["NOR"]
        # NOR: opposite of OR
        assert y == [1, 0, 0, 0]


class TestTrainingFunction:
    """Test the training function."""

    def test_train_and_gate(self) -> None:
        """Test training on AND gate - should converge."""
        w1, w2, bias, converged = main.train("AND", learning_rate=0.1, max_epochs=100)
        assert converged, "AND gate training should converge"
        assert isinstance(w1, float)
        assert isinstance(w2, float)
        assert isinstance(bias, float)

    def test_train_or_gate(self) -> None:
        """Test training on OR gate - should converge."""
        w1, w2, bias, converged = main.train("OR", learning_rate=0.1, max_epochs=100)
        assert converged, "OR gate training should converge"

    def test_train_nand_gate(self) -> None:
        """Test training on NAND gate - should converge."""
        w1, w2, bias, converged = main.train("NAND", learning_rate=0.1, max_epochs=100)
        assert converged, "NAND gate training should converge"

    def test_train_nor_gate(self) -> None:
        """Test training on NOR gate - should converge."""
        w1, w2, bias, converged = main.train("NOR", learning_rate=0.1, max_epochs=100)
        assert converged, "NOR gate training should converge"

    def test_train_xor_gate(self) -> None:
        """Test training on XOR gate - should NOT converge."""
        w1, w2, bias, converged = main.train("XOR", learning_rate=0.1, max_epochs=50)
        assert not converged, "XOR gate training should NOT converge"

    def test_train_invalid_gate(self) -> None:
        """Test training with invalid gate type raises error."""
        try:
            main.train("INVALID", learning_rate=0.1, max_epochs=10)
            assert False, "Should raise ValueError for invalid gate"
        except ValueError as e:
            assert "Unknown gate type" in str(e)

    def test_train_high_learning_rate(self) -> None:
        """Test training with higher learning rate."""
        w1, w2, bias, converged = main.train("AND", learning_rate=0.5, max_epochs=50)
        assert converged, "AND gate should converge with learning_rate=0.5"

    def test_train_deterministic(self) -> None:
        """Test that training is deterministic (same results on same inputs)."""
        w1_a, w2_a, bias_a, conv_a = main.train(
            "AND", learning_rate=0.1, max_epochs=100
        )
        w1_b, w2_b, bias_b, conv_b = main.train(
            "AND", learning_rate=0.1, max_epochs=100
        )

        # Results should be identical
        assert w1_a == w1_b
        assert w2_a == w2_b
        assert bias_a == bias_b
        assert conv_a == conv_b


class TestPredictionsAfterTraining:
    """Test that trained models make correct predictions."""

    def test_and_predictions(self) -> None:
        """Test AND gate predictions after training."""
        w1, w2, bias, _ = main.train("AND", learning_rate=0.1, max_epochs=100)
        x1, x2, y = main.LOGIC_GATES["AND"]

        # Check all training samples
        for i in range(4):
            pred = main.predict(x1[i], x2[i], w1, w2, bias)
            assert pred == y[i], f"AND gate prediction mismatch at sample {i}"

    def test_or_predictions(self) -> None:
        """Test OR gate predictions after training."""
        w1, w2, bias, _ = main.train("OR", learning_rate=0.1, max_epochs=100)
        x1, x2, y = main.LOGIC_GATES["OR"]

        for i in range(4):
            pred = main.predict(x1[i], x2[i], w1, w2, bias)
            assert pred == y[i], f"OR gate prediction mismatch at sample {i}"

    def test_nand_predictions(self) -> None:
        """Test NAND gate predictions after training."""
        w1, w2, bias, _ = main.train("NAND", learning_rate=0.1, max_epochs=100)
        x1, x2, y = main.LOGIC_GATES["NAND"]

        for i in range(4):
            pred = main.predict(x1[i], x2[i], w1, w2, bias)
            assert pred == y[i], f"NAND gate prediction mismatch at sample {i}"


class TestTestFunction:
    """Test the test/evaluation function."""

    def test_function_runs(self) -> None:
        """Test that the test function runs without error."""
        w1, w2, bias, _ = main.train("AND", learning_rate=0.1, max_epochs=100)
        test_data = main.LOGIC_GATES["AND"]

        # Should not raise an exception
        main.test(w1, w2, bias, test_data, "test set")

    def test_function_perfect_accuracy(self) -> None:
        """Test that trained model achieves perfect accuracy on training data."""
        w1, w2, bias, _ = main.train("OR", learning_rate=0.1, max_epochs=100)
        test_data = main.LOGIC_GATES["OR"]
        x1, x2, y = test_data

        # Count correct predictions
        correct = 0
        for i in range(len(x1)):
            pred = main.predict(x1[i], x2[i], w1, w2, bias)
            if pred == y[i]:
                correct += 1

        assert correct == len(y), "Trained model should have 100% accuracy"
