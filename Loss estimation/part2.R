Sys.setenv(RETICULATE_PYTHON = "C:\\Users\\Teodora\\miniconda3\\envs\\ids\\python.exe")

library(reticulate) # to use Python in R
pickle <- import("pickle")
pandas <- import("pandas")

# Load the Pickle file
data <- pandas$read_pickle("all_model_results.pkl")

# Convert to R data frame if necessary
df <- as.data.frame(data)
