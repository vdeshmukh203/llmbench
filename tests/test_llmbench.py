import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def test_import():
    import llmbench as lb
    assert hasattr(lb, 'BenchmarkRunner')

def test_rouge_l():
    import llmbench as lb
    assert lb.rouge_l('the cat sat', 'the cat sat') == 1.0

def test_exact_match():
    import llmbench as lb
    assert lb.exact_match('hello', 'hello') == 1.0
    assert lb.exact_match('hello', 'world') == 0.0

def test_approx_tokens():
    import llmbench as lb
    n = lb._approx_tokens('hello world')
    assert isinstance(n, int) and n > 0
