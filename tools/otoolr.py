#!/usr/bin/env python2
import subprocess

def otool(s):
    o = subprocess.Popen(['/usr/bin/otool', '-L', s], stdout=subprocess.PIPE)
    for l in o.stdout:
        if l[0] == '\t':
            yield l.split(' ', 1)[0][1:]

need = set(['/Volumes/Data/OP/WATCH/_vcv_rack/Rack___VCVRack_testing/plugins/DSW-Vision/plugin.dylib'])
done = set()

while need:
    needed = set(need)
    need = set()
    for f in needed:
        need.update(otool(f))
    done.update(needed)
    need.difference_update(done)

for f in sorted(done):
    print f