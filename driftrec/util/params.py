import warnings
import argparse


def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
        groups[group.title] = group_dict
    return groups


def required_length(nmin):
    # https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values):
                msg='argument "{f}" requires at least {nmin} arguments'.format(
                    f=self.dest,nmin=nmin)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength
