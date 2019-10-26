import inspect
import os
import json

import defenses
from defenses import Defense, DEFENSE_MAP


DOCTXT = 'doctxt'
TAGLINE = 'tagline'


def _get_signature(defense_class):
  parameters = list()

  sig = inspect.signature(defense_class.apply)
  for key, parameter in sig.parameters.items():
    if key not in ['image']:
      parameters.append((key, type(parameter.default), parameter.default))

  return parameters


def _process_doctxt(doctxt):
  for _ in range(4):
    doctxt = doctxt.strip()
    doctxt = doctxt.strip('\n')
  return doctxt


def main():

  input_schema = {'functions': []}
  output_schema = {'functions': []}

  input_schema[DOCTXT] = 'SHIELD contains image preprocessing techniques that can remove adversarial perturbations.'
  input_schema[TAGLINE] = 'Fast, Practical Defense for Deep Learning'

  for name, defense in DEFENSE_MAP.items():
    fn_input_schema = {
      'name': name,
      'extensions': [
        {'extension': 'jpg'}]}

    fn_output_schema = {
      'name': name,
      'output_tags': [
        {'name': 'mlsploit-visualize', 'type': 'str'}],
      'has_modified_files': True,
      'has_extra_files': False}

    fn_input_schema[DOCTXT] = \
      _process_doctxt(defense.doctxt)
    fn_doctxts = defense.option_doctxts

    options = []
    for (option_name,
          option_type,
          option_default) in _get_signature(defense):

      option_type = option_type.__name__
      opt = {'name': option_name,
              'type': option_type,
              'default': option_default,
              'required': True}
      opt[DOCTXT] = _process_doctxt(
        fn_doctxts[option_name])

      options.append(opt)

    fn_input_schema['options'] = options
    input_schema['functions'].append(fn_input_schema)
    output_schema['functions'].append(fn_output_schema)

  with open('input.schema', 'w') as f:
    json.dump(input_schema, f, indent=2)

  with open('output.schema', 'w') as f:
    json.dump(output_schema, f, indent=2)

if __name__ == "__main__":
  main()
