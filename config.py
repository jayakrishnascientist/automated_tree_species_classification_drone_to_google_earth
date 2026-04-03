class Config:

    # INPUT
    # Update this to the folder containing your multiple ortho images
    ORTHO_PATH = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/ortho"  # <-- new path
    WORKDIR = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/output"

    # DETECTREE
    DETECTREE_MODEL = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/250711_tropical_closed_canopy.pth"
    

    TILE_SIZE = 10
    BUFFER = 10
    IOU_THRESHOLD = 0.9
    CONF_THRESHOLD = 0.85

    # FEATURES + CLUSTERING
    STEP1_OUTPUT = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/output/step1_output"

    MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    PCA_COMPONENTS = 50

    K_LIST = [2, 4, 6, 8, 10]
    COPY_TO_CLUSTER_FOLDERS = True

    # SPECIES
    CHOSEN_K = 2
    STEP2_OUTPUT = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/output/step2_output"

    # VALIDATION
    GROUND_TRUTH_CSV = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/labels"
    STEP3_VALIDATION_OUTPUT = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/output/step3_output"

    # KMZ
    STEP4_OUTPUT = "/Users/jayakrishna/Desktop/untitled folder/end/tree_project/output/step4_output"
    SOURCE_EPSG = 32643

    COLOR_PALETTE = [
        "990000ff",
        "9900ff00",
        "99ff0000",
        "9900ffff",
        "99ff00ff",
        "99ff8800",
        "9900ffff",
        "99ffffff",
    ]