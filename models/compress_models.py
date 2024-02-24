import shutil

shutil.make_archive("models/lyft_models", 'zip', "models/lyft")
shutil.rmtree("models/lyft")
shutil.make_archive("models/uber_models", 'zip', "models/uber")
shutil.rmtree("models/uber")