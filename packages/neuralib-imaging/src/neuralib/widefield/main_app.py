from argclz.commands import parse_command_args
from neuralib.widefield.align import NapariAlignmentOptions
from neuralib.widefield.preproc import PreprocessOptions

from .fft_view import WideFieldFFTViewOption


def main():
    parsers = dict(
        preproc=PreprocessOptions,
        align=NapariAlignmentOptions,
        fft=WideFieldFFTViewOption
    )

    parse_command_args(
        parsers=parsers,
        description='widefield tools',
        usage="""
        Usage Examples:
        
        Preprocess of widefield image sequence:
        $ nl_widefield preproc ...

        View HSV map in FFT:
        $ nl_widefield fft <FILE>
        
        Alignment napari for image sequence
        $ nl_widefield align <FILE>
        
        """
    )


if __name__ == '__main__':
    main()
