DeepLabCut
=======================

- **Refer to API**: :mod:`neuralib.deeplabcut`

.. code-block:: python

    from neuralib.deeplabcut import read_dlc

    # load DeepLabCut output from .h5 OR .csv and its metadata (.pkl)
    dlc_df = read_dlc("/path/to/output.csv", meta_file="/path/to/meta.pkl")

    # access dataframe with all joints
    df = dlc_df.dataframe()

    # list available joints
    print("Joints:", dlc_df.joints)

    # access metadata
    print("Frames per second:", dlc_df.fps)
    print("Total frames:", dlc_df.nframes)

    # access a specific joint's data
    nose_df = dlc_df.get_joint("Nose").dataframe()
    print(nose_df)
