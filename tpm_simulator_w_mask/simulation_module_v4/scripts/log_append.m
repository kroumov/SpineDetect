function log_append(log_path, fmt, varargin)
% LOG_APPEND  Append formatted message to log file.
%
% log_append(log_path, fmt, varargin)
% If log_path is empty, no-op. Otherwise appends sprintf(fmt, varargin{:}) to file.

if isempty(log_path) || ~ischar(log_path)
  return;
end
fid = fopen(log_path, 'a');
if fid < 0
  return;
end
try
  fprintf(fid, fmt, varargin{:});
catch
end
fclose(fid);
end
