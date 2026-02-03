Facemap
===================

- **Refer to API**: :mod:`neuralib.facemap`

.. code-block:: python

    from neuralib.facemap import read_facemap

    # load a Facemap result directory
    result = read_facemap("/path/to/facemap/output")

    # check if keypoint tracking data is available
    if result.with_keypoint:

        # list available keypoints
        print("keypoints:", result.keypoints)

        # get data for a single keypoint
        df_eye = result.get("eye(back)").dataframe()
        print(df_eye)

        # get multiple keypoints and convert to z-scored coordinates
        df = result.get("eye(back)", "mouth").to_zscore()
        print(df)

    # access pupil tracking data
    pupil_area = result.get_pupil_area()
    pupil_com = result.get_pupil_center_of_mass()
