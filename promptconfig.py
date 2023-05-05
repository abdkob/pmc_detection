import yaml

# Prompt user for input
model_path = input("Enter the path to the ilastik model file: ")
ilastik_exec_path = input("Enter the path to the ilastik executable file: ")
log_file_path = input("Enter the path to the input log file: ")
data_dir_path = input("Enter the path to the directory containing the input data: ")
output_dir_path = input("Enter the path to the output directory: ")
sm50_radius = input("Enter the radius values for the sm50 gene (comma-separated list of three values): ")
pks2_radius = input("Enter the radius values for the pks2 gene (comma-separated list of three values): ")

# Split radius values into a list
sm50_radius = [int(x) for x in sm50_radius.split(",")]
pks2_radius = [int(x) for x in pks2_radius.split(",")]

# Load configuration file template
with open("config_template.yml", "r") as f:
    config_template = yaml.safe_load(f)

# Substitute user input into configuration file template
config = yaml.safe_load(str(config_template).format(
    model_path=model_path,
    ilastik_exec_path=ilastik_exec_path,
    log_file_path=log_file_path,
    data_dir_path=data_dir_path,
    output_dir_path=output_dir_path,
    sm50_radius=sm50_radius,
    pks2_radius=pks2_radius
))

# Print the resulting configuration
print("Configuration:")
print(yaml.dump(config))

# Run ilastik with the new configuration
# ...
