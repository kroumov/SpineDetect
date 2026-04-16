% INSTALL_MATLAB_ENGINE_FROM_MATLAB  Install MATLAB Engine for Python from MATLAB.
%
% Run in MATLAB: install_matlab_engine_from_matlab
%
% Must install from MATLAB engines\python dir; copying elsewhere causes "corrupted" error.
% Tries pip install from the original path first.

eng_src = fullfile(matlabroot, 'extern', 'engines', 'python');
if ~isfolder(eng_src)
  error('MATLAB Engine dir not found: %s', eng_src);
end

fprintf('Installing from MATLAB dir (recommended):\n');
try_cmds = {
  sprintf('cd /d "%s" && pip install . --user', eng_src)
  sprintf('cd /d "%s" && python -m pip install . --user', eng_src)
  sprintf('cd /d "%s" && py -3 -m pip install . --user', eng_src)
};
ok = false;
for i = 1:numel(try_cmds)
  cmd = try_cmds{i};
  fprintf('Try (%d/%d): cd ... && pip install . --user\n', i, numel(try_cmds));
  [st, msg] = system(cmd);
  disp(msg);
  if st == 0
    ok = true;
    break;
  end
  fprintf(2, 'Failed, trying next...\n\n');
end

if ok
  fprintf('Done. Verify: python -c "import matlab.engine; print(''OK'')"\n');
  return;
end

fprintf(2, '\nIf "access denied", run cmd as Administrator:\n\n');
fprintf('  cd /d "%s"\n', eng_src);
fprintf('  pip install . --user\n\n');
fprintf('Install from original path to pass "corrupted" check.\n');
