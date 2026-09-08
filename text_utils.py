# 'vector string constant width'
def vscw(L: list): # TODO: Fix issue in which signs change the width of numbers
    if L is None: return "None" # Simple
    s = "("
    for i in L: s += "{:.4f}".format(i) + ", "
    return s[:-2] + ")"
