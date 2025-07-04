# install.packages("jsonlite") 
#install.packages('pmlbr')
library(jsonlite)
library(pmlbr)

#Airfoil-----------------------------------------------------------------------------------------------
file_path <- "C:/Users/rowan/Downloads/airfoil+self+noise/airfoil_self_noise.dat"  # <-- Change path here

# Define column names, 5 predictors and 1 target
column_names <- c("Predictor1", "Predictor2", "Predictor3",
                  "Predictor4", "Predictor5", "Target")

# Read the data
tryCatch({
  my_data <- read.table(file_path, header = FALSE, sep = "", col.names = column_names)
  
  # Display the first few rows to verify is preproccessing was done correctly
  print(head(my_data))
  
  # Check if exactly 6 columns were read
  if (ncol(my_data) != 6) {
    warning(paste("Warning: Expected 6 columns, but read", ncol(my_data), "columns. Check 'sep' or file content."))
  }
  
}, error = function(e) {
  cat("Error reading the file:", conditionMessage(e), "\n")
})

write.csv(my_data, "airfoil_table.csv", row.names = FALSE)

#Boston housing ---------------------------------------------------------------------------------------------

# Read the CSV file 
boston_data <- read.csv("C:/Users/rowan/Downloads/archive (8)/BostonHousing.csv") # <-- Change path here
names(boston_data)[ncol(boston_data)] <- "Target"

# View the first few rows to verify
head(boston_data)

write.csv(boston_data, "boston_table.csv", row.names = FALSE)

#Communities ------------------------------------------------------------------------------------------------
file_path <- "C:/Users/rowan/Downloads/communities+and+crime/communities.data" # <-- Change path here

# Read the data
tryCatch({
  raw_data <- read.csv(file_path,
                       header = FALSE,
                       sep = ",",
                       na.strings = "?",
                       stringsAsFactors = FALSE)
  
  cat("Original dimensions (rows x columns):", dim(raw_data)[1], "x", dim(raw_data)[2], "\n")
  
  # Check if we have enough columns to remove the first 5
  if (ncol(raw_data) < 5) {
    stop(paste("Error: The file has only", ncol(raw_data), "columns. Cannot remove the first 5 specified columns."))
  }
  if (ncol(raw_data) != 128) {
    warning(paste("Warning: Expected 128 columns for the full Communities and Crime dataset, but read",
                  ncol(raw_data), "columns. Proceeding with column removal based on position."))
  }
  
  # The columns to delete are:
  # 1st: state
  # 2nd: county
  # 3rd: community
  # 4th: communityname
  # 5th: fold
  communities_features_target <- raw_data[, -(1:5)] # Keep all rows, remove columns 1 through 5
  
  cat("\nRemoved the first 5 columns (state, county, community, communityname, fold).\n")
  cat("New dimensions (rows x columns):", dim(communities_features_target)[1], "x", dim(communities_features_target)[2], "\n")
  # Expected new number of columns: 128 - 5 = 123
  
  # Assigning Generic Column Names to the Remaining Data
  num_remaining_cols <- ncol(communities_features_target)
  if (num_remaining_cols > 0) {
    # Create generic names: Feature1, Feature2, ..., FeatureN-1, Target
    feature_names <- paste0("Feature", 1:(num_remaining_cols - 1))
    all_new_colnames <- c(feature_names, "Target")
    colnames(communities_features_target) <- all_new_colnames
    cat("\nAssigned generic column names to the remaining", num_remaining_cols, "columns.\n")
  }
  
  # Display some info to verify the result
  cat("\nFirst few rows and first few columns of the processed data:\n")
  print(head(communities_features_target[, 1:min(6, ncol(communities_features_target))])) 
  
}, error = function(e) {
  cat("Error during processing:", conditionMessage(e), "\n")
})

write.csv(communities_features_target, "communities_table.csv", row.names=FALSE)

# Graduate Admission --------------------------------------------------------------------------------------
admission_data = read.csv("https://github.com/ybifoundation/Dataset/raw/main/Admission%20Chance.csv") # <-- Change path here
names(admission_data)[ncol(admission_data)] <- "Target"

# View the first few rows to verify
head(admission_data)

# Remove index column
admission_data = admission_data[,-1]

write.csv(admission_data, "admission_table.csv", row.names = FALSE)

# Ozone -----------------------------------------------------------------------------------------
#install.packages("missMDA", dependencies = TRUE)

Sys.which("make")
library(missMDA)
data(ozone) 

set.seed(123) # For reproducibility

# Fill in the missing values using function from the package self
ncp_est <- estim_ncpFAMD(ozone, ncp.max = 5)$ncp
print(ncp_est)
imputed <- imputeFAMD(ozone, ncp = ncp_est)
ozone_complete <- imputed$completeObs

cat("First few rows of the Ozone dataset:\n")
print(head(ozone_complete))

cat("\nDimensions of the Ozone dataset (rows x columns):\n")
print(dim(ozone_complete))

names(ozone_complete)[1] <- "Target"

# Print to check if preperation was succesful
print(ozone_complete)

write.csv(ozone_complete, "ozone_table.csv", row.names = FALSE)

# Real estate --------------------------------------------------------------------------
library(readxl)
real_data = read_excel("C:/Users/rowan/Downloads/real+estate+valuation+data+set/Real estate valuation data set.xlsx") # <-- Change path here
names(real_data)[ncol(real_data)] <- "Target"

# View the first few rows to verify
head(real_data)

# Remove index column
real_data = real_data[,-1]

write.csv(real_data, "real_table.csv", row.names = FALSE)

# Thermography -------------------------------------------------------------------------------
therm_data = read.csv("C:/Users/rowan/Documents/themograpy_table.csv") # <-- Change path here

# Target = AveOralF
therm_data_F = therm_data[,-ncol(therm_data)]
names(therm_data_F)[ncol(therm_data_F)] <- "Target"
# Print to check if preperation was succesful
print(colnames(therm_data_F))
print(therm_data_F[,ncol(therm_data_F)])

# Target = AveOralM
therm_data_M = therm_data[,-(ncol(therm_data)-1)]
names(therm_data_M)[ncol(therm_data_M)] <- "Target"
# Print to check if preperation was succesful
print(colnames(therm_data_M))
print(therm_data_M[ncol(therm_data_M)])

write.csv(therm_data_F, "thermF_table.csv", row.names = FALSE)
write.csv(therm_data_M, "thermM_table.csv", row.names = FALSE)

# Residential ---------------------------------------------------------------------------------
residential_data = read_excel("C:/Users/rowan/Downloads/residential+building+data+set (1)/Residential-Building-Data-Set.xlsx")

# Only use the necessary columns and rows
residential_data = residential_data[2:nrow(residential_data),5:ncol(residential_data)]

# Target = sales
residential_data_sales = residential_data[,-ncol(residential_data)]
names(residential_data_sales)[ncol(residential_data_sales)] <- "Target"
# Print to check if preperation was succesful
print(colnames(residential_data_sales))
print(residential_data_sales[,ncol(residential_data_sales)])

# Target = costs
residential_data_costs = residential_data[,-(ncol(residential_data)-1)]
names(residential_data_costs)[ncol(residential_data_costs)] <- "Target"
# Print to check if preperation was succesful
print(colnames(residential_data_costs))
print(residential_data_costs[ncol(residential_data_costs)])

write.csv(residential_data_sales, "ressales_table.csv", row.names = FALSE)
write.csv(residential_data_costs, "rescosts_table.csv", row.names = FALSE)

# Slump test ---------------------------------------------------------------------------------
slump_data = read.csv("C:/Users/rowan/Downloads/concrete+slump+test (1)/slump_test.data") # <-- Change path here

slump_data = slump_data[,-1]
print(slump_data)      

# Target = slump
slump_slump = slump_data[,-((ncol(slump_data)-1) : ncol(slump_data))]
names(slump_slump)[ncol(slump_slump)] = "Target"
# Print to check if preperation was succesful
print(slump_slump)

# Target = flow
slump_flow = slump_data[,-ncol(slump_data)]
slump_flow = slump_flow[,-(ncol(slump_flow)-1)]
names(slump_flow)[ncol(slump_flow)] = "Target"
# Print to check if preperation was succesful
print(slump_flow)

# Target = strength
slump_strength = slump_data[,-((ncol(slump_data)-2) : (ncol(slump_data)-1))]
names(slump_strength)[ncol(slump_strength)] = "Target"
# Print to check if preperation was succesful
print(slump_strength)

write.csv(slump_slump, "slumpSL_table.csv", row.names = FALSE)
write.csv(slump_flow, "slumpFL_table.csv", row.names = FALSE)
write.csv(slump_strength, "slumpST_table.csv", row.names = FALSE)

# Tecator ----------------------------------------------------------------------------------
tecator_data_raw <- fetch_data('505_tecator')

# Use only the absorbance channels and Fat as target
tecator_data_exp <- tecator_data_raw[, 1 : 100]
tecator_data_targ <- tecator_data_raw$target

# Make them together in a dataframe
tecator_data <- data.frame(tecator_data_exp, Target = tecator_data_targ)

# Print to check if preperation was succesful
print(tecator_data)

write.csv(tecator_data, "tecator_table.csv", row.names = FALSE)

# Geographical Origin of Music --------------------------------------------------------------
music_data_raw <- read.csv("C:/Users/rowan/Downloads/geographical+original+of+music (1)/Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt", header = FALSE)
print(paste("Loaded data with dimensions:", paste(dim(music_data_raw), collapse = " x "))) # Should be 1059 x 118

# The first 116 columns are the audio features
features_orig <- music_data_raw[, 1:116]

# The 117th column is the target variable: Latitude
latitude_target <- music_data_raw[, 117]

# The 118th column (Longitude) will be discarded to avoid data leakage
longitude_discarded <- music_data_raw[, 118]

# Remove duplicate columns
print(paste("Original feature count:", ncol(features_orig))) # Should be 116
is_duplicate_col <- duplicated(as.list(features_orig))
features_clean <- features_orig[, !is_duplicate_col]
print(paste("Final unique feature count:", ncol(features_clean))) # Should be 72

# Combine the 72 unique features with the Latitude target column.
music_data_final <- data.frame(features_clean, Target = latitude_target)

# Verify the Final Dataset
print(paste("Final dimensions:", paste(dim(music_data_final), collapse = " x "))) # Should be 1059 x 73
head(music_data_final[, c(1:5, ncol(music_data_final))])

write.csv(music_data_final, "music_table.csv", row.names = FALSE)
