"""Test module"""


def check_labels():
    """Check CIFAR-100 labels"""
    with open("cifar100.txt", "r") as f:
        text = f.read()
        labels = text.split("\n")
    return labels


def test_1():
    """Test amount of elements"""
    labels = check_labels()
    assert len(labels) == 100


def test_2():
    """Test first element"""
    labels = check_labels()
    assert labels[0] == 'apple'


def test_3():
    """Test 50'th element"""
    labels = check_labels()
    assert labels[49] == 'mountain'