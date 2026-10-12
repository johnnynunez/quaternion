"""Independent CPU numerical witnesses; no Torch, no tolerance relaxation."""
import math
import os
import numpy as np
import pytest

KIND = os.environ.get('AUDIT_KIND', 'quaternion')

@pytest.mark.skipif(KIND != 'quaternion', reason='other distribution')
def test_quaternion_hamilton_composition_norm_and_rotation():
    import quaternion as q
    a = q.quaternion(1, 2, -3, 4)
    b = q.quaternion(-2, 5, 1, -1)
    av, bv = [1, 2, -3, 4], [-2, 5, 1, -1]
    w,x,y,z = av; W,X,Y,Z = bv
    expected = [w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
                w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W]
    np.testing.assert_array_equal(q.as_float_array(a*b), expected)
    assert abs(a) == math.sqrt(sum(v*v for v in av))
    np.testing.assert_allclose(q.as_float_array(a*a.inverse()), [1,0,0,0], atol=3e-16, rtol=3e-16)
    np.testing.assert_allclose(abs(a*b), abs(a)*abs(b), rtol=3e-16)
    rz = q.quaternion(math.cos(math.pi/4), 0, 0, math.sin(math.pi/4))
    rx = q.quaternion(math.cos(math.pi/4), math.sin(math.pi/4), 0, 0)
    v = np.array([1.,2.,3.])
    np.testing.assert_allclose(q.rotate_vectors(rz,v), [-2,1,3], atol=1e-15)
    np.testing.assert_allclose(q.rotate_vectors(rz*rx,v), [3,1,2], atol=2e-15)
    np.testing.assert_allclose(q.rotate_vectors(-rz,v), q.rotate_vectors(rz,v), atol=1e-15)
    np.testing.assert_allclose(q.as_rotation_matrix(rz).T @ q.as_rotation_matrix(rz), np.eye(3), atol=1e-15)

@pytest.mark.skipif(KIND != 'quaternion', reason='other distribution')
def test_quaternion_nan_zero_and_wrong_inputs():
    import quaternion as q
    a = q.quaternion(float('nan'),0,0,0)
    assert np.isnan(a)
    assert not np.isfinite(a)
    assert a != a
    assert abs(q.quaternion(0,0,0,0)) == 0
    with pytest.raises((ValueError, TypeError)):
        q.as_quat_array(np.ones(3))
    with pytest.raises((ValueError, TypeError)):
        q.rotate_vectors(q.one, np.ones(2))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(q.as_float_array(q.i*q.j), [0,0,0,-1], atol=0)

@pytest.mark.skipif(KIND != 'scikit-image', reason='other distribution')
def test_image_gaussian_against_scalar_convolution():
    from skimage.filters import gaussian
    rng = np.random.default_rng(721)
    a = rng.normal(size=(9,11))
    radius = 4
    kernel = [math.exp(-0.5*i*i) for i in range(-radius, radius+1)]
    kernel = [v/sum(kernel) for v in kernel]
    expected = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            expected[i,j] = sum(kernel[u+radius]*kernel[v+radius]*a[i+u,j+v]
                for u in range(-radius,radius+1) for v in range(-radius,radius+1)
                if 0 <= i+u < a.shape[0] and 0 <= j+v < a.shape[1])
    actual = gaussian(a, 1., mode='constant', cval=0, truncate=4., preserve_range=True)
    np.testing.assert_allclose(actual, expected, atol=4e-16, rtol=2e-14)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, expected+0.1, atol=4e-16, rtol=2e-14)
    assert np.isnan(gaussian(np.full((3,3), np.nan))).all()
    with pytest.raises(ValueError):
        gaussian(a, sigma=-1)

@pytest.mark.skipif(KIND != 'scikit-image', reason='other distribution')
def test_image_morphology_and_components_scalar_oracles():
    from skimage.morphology import binary_dilation, binary_erosion
    from skimage.measure import label
    a = np.zeros((8,9),dtype=bool)
    a[1:3,1:3]=True; a[5:7,6:8]=True; a[3,4]=True
    fp = np.ones((3,3),dtype=bool)
    dil = np.zeros_like(a); ero = np.zeros_like(a)
    for i in range(8):
        for j in range(9):
            vals=[a[x,y] for x in range(max(0,i-1),min(8,i+2)) for y in range(max(0,j-1),min(9,j+2))]
            dil[i,j] = any(vals); ero[i,j] = all(vals)
    np.testing.assert_array_equal(binary_dilation(a,fp),dil)
    np.testing.assert_array_equal(binary_erosion(a,fp),ero)
    expected = np.zeros(a.shape,dtype=int); n=0
    for i,j in np.ndindex(a.shape):
        if not a[i,j] or expected[i,j]: continue
        n+=1; todo=[(i,j)]; expected[i,j]=n
        while todo:
            x,y=todo.pop()
            for u,v in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0<=u<8 and 0<=v<9 and a[u,v] and not expected[u,v]:
                    expected[u,v]=n; todo.append((u,v))
    np.testing.assert_array_equal(label(a,connectivity=1),expected)
    with pytest.raises(ValueError): label(a,connectivity=0)

@pytest.mark.skipif(KIND != 'scikit-image', reason='other distribution')
def test_image_resize_independent_bilinear():
    from skimage.transform import resize
    a = np.array([[0.,2.],[4.,8.]])
    expected=np.empty((4,4))
    for i,j in np.ndindex(expected.shape):
        y=max(0.,min(1.,(i+0.5)/2-0.5)); x=max(0.,min(1.,(j+0.5)/2-0.5))
        expected[i,j]=(1-y)*((1-x)*a[0,0]+x*a[0,1])+y*((1-x)*a[1,0]+x*a[1,1])
    np.testing.assert_allclose(resize(a,(4,4),order=1,mode='edge',anti_aliasing=False,preserve_range=True),expected,atol=2e-15,rtol=2e-15)
    np.testing.assert_array_equal(resize(a,(4,4),order=0,anti_aliasing=False,preserve_range=True),np.repeat(np.repeat(a,2,0),2,1))
    with pytest.raises(ValueError): resize(a,(4,4),order=9)
