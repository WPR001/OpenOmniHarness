"""The built-in guard, always on. A command that names a path outside the sandbox, climbs out with '..', or uses
network, process, package or system tools is refused with the reason shown. Every command is logged in the record."""
import re

GUARD = [
    (re.compile(r'(?<![\w.)\]}-])/(mnt|home|root|etc|usr|opt|var|proc|sys|tmp|data|srv|bin|sbin|lib|lib64|boot|run|'
                r'snap|media|scratch|dev/(?!null\b))'),
     'absolute paths are not available; write and read files with relative names inside the sandbox'),
    (re.compile(r'(^|[\s/"\'=(])\.\.(/|\s|$|["\')])'), "'..' is not allowed; stay inside the sandbox"),
    (re.compile(r'(^|[\s;&|(`=])~(/|$)|\bcd\s+~\s*($|[;&|])'), 'the home directory is not available'),
    (re.compile(r'\b(curl|wget|ssh|scp|rsync|nc|ncat|telnet|git|pip3?|conda|apt|apt-get|sudo|su|docker|srun|sbatch|'
                r'scancel|squeue|nvidia-smi|kill|pkill|killall|reboot|shutdown|crontab|mount|chmod|chown)\b'),
     'network, package, process and system commands are not available'),
    (re.compile(r'\b(socket|urllib|requests|http\.client|subprocess|os\.system|shutil\.rmtree)\b'),
     'network and process modules are not available'),
    (re.compile(r'\bffmpeg\b.*(\s-vn\b|\.(wav|mp3|aac|flac|m4a)\b)'), 'there is no audio in this video; use asr'),
]
