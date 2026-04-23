function A = array_SubSubTest(A, idx, val, sc)
% Pure MATLAB fallback when MEX not compiled. NAOMi provides .cpp only.
% A(idx) = val * sc (assignment)
idx = int32(idx(:));
val = single(val(:)) * single(sc);
for ii = 1:numel(idx)
  A(idx(ii)) = val(ii);
end
end
