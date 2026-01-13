from bias_analysis.contingency import Contingency2x2

def test_table_shape():
    t = Contingency2x2(a=1,b=2,c=3,d=4)
    arr = t.as_array()
    assert arr.shape == (2,2)
    assert arr[0,0] == 1
