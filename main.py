import os

from mlsploit_local import Job

from data import (
    build_image_dataset,
    get_or_create_dataset,
    process_image,
    recreate_image,
)
from defenses import DEFENSE_MAP


def main():
    # Initialize the job, which will
    # load and verify all input parameters
    Job.initialize()

    defense_name = Job.function
    defense_options = dict(Job.options)
    defense_class = DEFENSE_MAP[defense_name]

    input_file_paths = list(map(lambda f: f.path, Job.input_files))
    input_dataset, is_temp_dataset = get_or_create_dataset(input_file_paths)

    output_dataset = build_image_dataset(
        Job.make_output_filepath(input_dataset.path.name)
    )

    for item in input_dataset:
        input_image = recreate_image(item.data)

        output_image = defense_class.apply(input_image, **defense_options)
        output_image_arr = process_image(output_image)

        output_dataset.add_item(
            name=item.name, data=output_image_arr, label=item.label, prediction=-1
        )

    output_item = output_dataset[0]
    output_image = recreate_image(output_item.data)
    output_image_path = Job.make_output_filepath(output_item.name)
    output_image.save(output_image_path)

    Job.add_output_file(str(output_dataset.path), is_extra=True)
    Job.add_output_file(
        output_image_path, is_modified=True, tags={"mlsploit-visualize": "image"}
    )

    Job.commit_output()

    if is_temp_dataset:
        os.remove(input_dataset.path)


if __name__ == "__main__":
    main()
