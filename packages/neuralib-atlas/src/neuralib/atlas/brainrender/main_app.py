from argclz.commands import parse_command_args

from .core import BrainRenderCLI
from .probe import ProbeRenderCLI
from .roi import RoiRenderCLI


def main():
    parsers = dict(
        area=BrainRenderCLI,
        roi=RoiRenderCLI,
        probe=ProbeRenderCLI
    )

    parse_command_args(
        parsers=parsers,
        description='BrainRender CLI Options',
        usage="""
        Usage Examples:
    
        Render a brain region:
        $ nl_brainrender area -R <REGION, ...>
        $ nl_brainrender area -R VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor
    
        Render a region of interest (ROI) from a file:
        $ nl_brainrender roi -F <FILE>
    
        Render a probe placement from a file with depth specification:
        $ nl_brainrender probe -F <FILE> --depth <DEPTH>
        """
    )


if __name__ == '__main__':
    main()
