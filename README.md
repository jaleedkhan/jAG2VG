# jAG2VG
Formatting Action Genome dataset in Visual Genome's format

The [jAG2VG.ipynb notebook](https://github.com/jaleedkhan/jAG2VG/blob/master/jAG2VG.ipynb)
- downloads the [Action Genome (AG) dataset](actiongenome.org)
- extracts those frames from the AG videos which are annotated in the dataset
- re-formats the AG annotations (pickle files) as JSON files in a format similar to the [Visual Genome (VG) dataset](https://visualgenome.org/)'s annotations
- provides the AG data in VG's format in a new folder 'AGinVGformat/':
  - 'VG_100K/' contains all the frames
  - The JSON files contain annotations (extracted from AG pickle) of objects, relationships, scene graphs, attributes, etc. for each frame

The VG dataset has been widely used in a number of computer vision and image understanding tasks including scene graph generation, image captioning, VQA etc. This code allows the use of AG for the same tasks. It has been tested for scene graph generation only.
