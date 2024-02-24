import shutil

# zip regression models for lyft and uber
shutil.make_archive("models/lyft_reg_models", 'zip', "models/lyft_reg_models")
shutil.make_archive("models/uber_reg_models", 'zip', "models/uber_reg_models")

# zip tree models for lyft and uber
shutil.make_archive("models/lyft_tree_models", 'zip', "models/lyft_tree_models")
shutil.make_archive("models/uber_tree_models", 'zip', "models/uber_tree_models")

# zip deep models for lyft and uber
shutil.make_archive("models/lyft_deep_models", 'zip', "models/lyft_deep_models")
shutil.make_archive("models/uber_deep_models", 'zip', "models/uber_deep_models")