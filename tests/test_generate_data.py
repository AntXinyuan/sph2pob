import utils.ext_import

from utils.generate_data import generate_ranges, generate_boxes
from utils.timer import Timer, print_run_time
from sphdet.bbox.box_formator import is_valid_boxes

def test_generate_boxes():
    bboxes = generate_boxes(10)
    assert bboxes.shape == (10, 4)

def test_generate_ranges():
    ranges = generate_ranges(30)
    assert ranges.shape == (6, 12, 2, 2)

    ranges = generate_ranges(30, flaten=True)
    assert ranges.shape == (6*12, 2, 2)

    ranges = generate_ranges(30, mode='diag')
    assert ranges.shape == (min(6, 12), 2, 2)

    ranges = generate_ranges(30, flaten=True, mode='diag')
    assert ranges.shape == (min(6, 12), 2, 2)

def test_boxes_validator():
    bboxes_good = generate_boxes(10, dtype='float')
    bboxes_bad  = generate_boxes(10, phi_range=(360, 370))
    assert is_valid_boxes(bboxes_good)
    assert is_valid_boxes(bboxes_bad, need_raise=True)

if __name__ == '__main__':
    test_generate_boxes()
    test_generate_ranges()
    test_boxes_validator()