function warn_append(log_path, fmt, varargin)
% WARN_APPEND  Print warning to terminal and append to log file.
% Used for fallback mechanisms so both terminal and log receive the warning.
%
% warn_append(log_path, fmt, varargin)
% Same signature as log_append; prints to stdout with 'WARN: ' prefix and to log.

msg = sprintf(fmt, varargin{:});
fprintf(1, 'WARN: %s', msg);
if ~isempty(log_path) && ischar(log_path)
  log_append(log_path, 'WARN: %s', msg);
end
end
