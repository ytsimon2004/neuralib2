from argclz.commands import parse_command_args
from neuralib.widefield.align import NapariAlignmentOptions
from neuralib.widefield.preproc import PreprocessOptions

from .fft_view import WideFieldFFTViewOption
from .transform import RegistrationOptions


def main():
    parse_command_args(
        parsers=dict(
            preproc=PreprocessOptions,
            align=NapariAlignmentOptions,
            fft=WideFieldFFTViewOption,
            trans=RegistrationOptions
        ),
        description='widefield tools',
        usage="""
        Usage Examples:
        
        Preprocess of widefield image sequence:
        $ nl_wfield preproc ...

        View HSV map in FFT:
        $ nl_wfield fft <FILE>
        
        Alignment napari for image sequence
        $ nl_wfield align <FILE>

        Widefield-to-dorsal-map registration GUI
        $ nl_wfield trans
        
        """
    )


if __name__ == '__main__':
    main()
