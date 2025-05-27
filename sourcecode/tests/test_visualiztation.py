import numpy as np
import pytest
from visualization import interactive_slice_viewer

def test_interactive_slice_viewer_slicing():
    """
    Test that interactive_slice_viewer correctly updates the image data
    when the slider value changes.
    """
    # Create a simple volume: depth=4, height=2, width=2
    cube = np.zeros((2, 2, 4))
    # Fill each slice with its index
    for z in range(4):
        cube[:, :, z] = z

    # Call viewer in test mode (no GUI)
    fig, slider, im = interactive_slice_viewer(cube, show=False)

    # Initial image should show middle slice: z=2
    initial = im.get_array().copy()
    assert np.all(initial == 2), f"Expected initial slice 2, got {initial}"

    # Move slider to slice 0
    slider.set_val(0)
    arr0 = im.get_array()
    assert np.all(arr0 == 0), f"Expected slice 0 after slider=0, got {arr0}"

    # Move slider to slice 3
    slider.set_val(3)
    arr3 = im.get_array()
    assert np.all(arr3 == 3), f"Expected slice 3 after slider=3, got {arr3}"

def test_invalid_volume_raises():
    """
    interactive_slice_viewer should reject non-3D arrays.
    """
    with pytest.raises(ValueError):
        interactive_slice_viewer(np.zeros((2, 2)), show=False)