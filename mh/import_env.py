import os
from subprocess import check_output as co

import numpy as np


def sco(s, split=True):
    try:
        res = co(s, shell=True).decode("utf-8")
    except:
        raise
    if split:
        return res.split("\n")[:-1]
    else:
        return res


def get_local_name(s, ext=".py"):
    u = s if "/" not in s else s.split("/")[-1]
    u = u.replace(ext + "x", "")
    return u.replace(ext, "")


def get_tracked_files():
    return set(sco('git ls-files | grep ".py$" | grep -v "__init__.py" | xargs -I {} readlink -f {}', True))


def get_subfolders(path, **kw):
    omissions = kw.get("omissions", [])
    inclusions = kw.get("inclusions", None)
    local = kw.get("local", True)
    ext = kw.get("ext", ".py")
    depth = kw.get("depth", 1)
    omissions = [path + "/%s" % e if "/" not in e else e for e in omissions]
    if inclusions != None:
        inclusions = [
            path + "/%s" % e if "/" not in e else e for e in inclusions
        ]
    try:
        cmd = r"find %s -maxdepth %d -mindepth %d -type d" % (
            path,
            depth,
            depth,
        )
        u = sco(cmd)
        u = [
            e
            for e in u
            if not e.split("/")[-1].startswith("__")
            and not e.split("/")[-1].startswith(".")
        ]
    except:
        u = []
    # if only_tracked:
    #     input(u)
    #     input(get_tracked_files())
    #     u = [d for d in u if d in get_tracked_files()]
    if len(omissions) > 0 or inclusions != None:
        if inclusions != None:
            [omissions.append(e) for e in u if e not in inclusions]
        u = [e for e in u if e not in omissions]
    if local:
        u = [get_local_name(e, ext) for e in u]
    return u


def get_local_modules(path, *, only_tracked=False, **kw):
    local = kw.get("local", True)
    ext = kw.get("ext", ".py")
    res = sco(
        r'find %s -mindepth 1 -maxdepth 1 -type f -name "*%s"' % (path, ext)
    )
    res2 = sco(
        r'find %s -mindepth 1 -maxdepth 1 -type f -name "*%sx"' % (path, ext)
    )
    [res.append(e) for e in res2]
    if only_tracked:
        res = [f for f in res if f in get_tracked_files()]
    res = [
        e
        for e in res
        if not (
            e.split("/")[-1].startswith(".") or e.split("/")[-1].startswith("_")
        )
    ]
    if local:
        res = [get_local_name(e) for e in res]
    return res


def init_modules(path, *, only_tracked=False, **kw):
    root = kw.get("root", False)
    unload = kw.get("unload", False)
    if root:
        local_modules = []
    else:
        local_modules = get_local_modules(path, only_tracked=only_tracked, **kw)
    subfolders = get_subfolders(path, only_tracked=only_tracked, **kw)
    [local_modules.append(e) for e in subfolders]
    s = "__all__ = [\n"
    for e in local_modules:
        s += '    "%s",\n' % e
    if s == "__all__ = [\n":
        s += "]\n"
    else:
        s = s[:-2]
        s += "\n]\n"
    if unload:
        v = (
            s.translate(str.maketrans("", "", ' \n"'))
            .split("[")[1]
            .split("]")[0]
            .split(",")
        )
        for e in v:
            s += "from .%s import *\n" % e
    #        s += 'from . import *'
    if True:
        # s = format_with_black(s, preview=False)
        filename = path + "/__init__.py"
        with open(filename, "w") as f:
            print('Write to "%s"\n"%s"' % (filename, s))
            f.write(s)

    global_subfolders = ["%s/%s" % (path, e) for e in subfolders]
    kw["root"] = False
    kw["inclusions"] = None
    for e in global_subfolders:
        init_modules(e, **kw)


def run_make_files(omissions=[]):
    ref_path = os.getcwd()
    omissions = [
        e if e.startswith("/") else "%s/%s" % (ref_path, e) for e in omissions
    ]
    make_files = sco('find %s -name "Makefile"' % ref_path, True)
    make_dir = np.array([e.replace("/Makefile", "") for e in make_files])
    omit = [np.any([ee in e for ee in omissions]) for e in make_dir]
    make_dir = [e for (i, e) in enumerate(make_dir) if not omit[i]]
    for d in make_dir:
        os.chdir(d)
        os.system("rm __init__.py")
        os.system("make clean")
        os.system("make")
    os.chdir(ref_path)
