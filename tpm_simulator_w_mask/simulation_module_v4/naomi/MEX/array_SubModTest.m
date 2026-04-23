function A = array_SubModTest(A, idx, val, sc)
% Pure MATLAB fallback when MEX not compiled. NAOMi provides .cpp only.
% A(idx) = A(idx) + val * sc (add)
idx = int32(idx(:));
val = single(val(:)) * single(sc);
for ii = 1:numel(idx)
  A(idx(ii)) = A(idx(ii)) + val(ii);
end
end
