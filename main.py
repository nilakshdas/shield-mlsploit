import os

from mlsploit import Job
from PIL import Image

from defenses import DEFENSE_MAP


# Initialize the job, which will
# load and verify all input parameters
Job.initialize()

defense_name = Job.function
defense_options = dict(Job.options)
defense_class = DEFENSE_MAP[defense_name]

input_file_paths = list(map(lambda f: f.path, Job.input_files))
input_file_path = input_file_paths[0]
input_file_name = os.path.basename(input_file_path)

image = Image.open(input_file_path)
output_image = defense_class.apply(image, **defense_options)

output_file_path = Job.make_output_filepath(input_file_name)
output_image.save(output_file_path)
Job.add_output_file(
    output_file_path, is_modified=True, tags={"mlsploit-visualize": "image"}
)
Job.commit_output()
