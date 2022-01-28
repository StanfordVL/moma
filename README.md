## Introduction
MOMA is a dataset dedicated to multi-object, multi-actor activity recognition. 

## Requirements

- Python 3.9+
- ffmpeg: For data processing.
- Torchvision 0.11.0+: For data loading and visualization.
- [distinctipy](https://github.com/alan-turing-institute/distinctipy): For data visualization. Install by running `pip install distinctipy`.

## Hierarchy
| Level | Concept                                              | Representation                                                 |
|-------|------------------------------------------------------|----------------------------------------------------------------|
| 1     | Activity                                             | Semantic label                                                 |
| 2     | Sub-activity                                         | Temporal boundary and semantic label                           |
| 3     | Higher-order interaction                             | Spatial-temporal scene graph                                   |
|       | └─&emsp;Entity                                       | Graph node w/ bounding box, instance label, and semantic label |
|       | &emsp;&emsp;├─&emsp;Actor                            | -                                                              |
|       | &emsp;&emsp;└─&emsp;Object                           | -                                                              |
|       | └─&emsp;Description                                  | Graph edge w/ semantic label                                   |
|       | &emsp;&emsp;└─&emsp;State                            | -                                                              |
|       | &emsp;&emsp;&emsp;&emsp;├─&emsp;Attribute            | Loop                                                           |
|       | &emsp;&emsp;&emsp;&emsp;└─&emsp;Relationship         | Directed edge                                                  |
|       | &emsp;&emsp;└─&emsp;Atomic action                    | -                                                              |
|       | &emsp;&emsp;&emsp;&emsp;├─&emsp;Intransitive action  | Loop                                                           |
|       | &emsp;&emsp;&emsp;&emsp;└─&emsp;Transitive action    | Directed edge                                                  |


## Annotations
In this version, we include:
- 1,411 activity instances from 20 activity classes.
- 15,436 sub-activity instances from 97 sub-activity classes.
- 156,790 higher-order interaction instances
- 599,843 actor bboxes, 98,325 actor instances
- 338,990 object bboxes, 46,034 object instances
- 412,914 intransitive action instances
- 38,666 transitive action instances
- 251,779 attribute instances
- 951,543 relationship instances

Below, we show the syntax of the MOMA annotations.
```json5
[
  {
    "file_name": str,
    "num_frames": int,
    "width": int,
    "height": int,
    "duration": float,

    // an activity
    "activity": {
      "id": str,
      "class_name": str,
      "start_time": int,
      "end_time": int,
      
      "sub_activities": [
        // a sub-activity
        {
          "id": str,
          "class_name": str,
          "start_time": int,
          "end_time": int,
          
          "higher_order_interactions": [
            // a higher-order interaction
            {
              "id": str,
              "time": int,
              
              "actors": [
                // an actor
                {
                  "id": str,
                  "class_name": str,
                  "bbox": [x, y, width, height]
                },
                ...
              ],
              
              "objects": [
                // an object
                {
                  "id": str,
                  "class_name": str,
                  "bbox": [x, y, width, height]
                },
                ...
              ],
              
              "relationships": [
                // a relationship
                {
                  "class_name": str,
                  "source_id": str,
                  "target_id": str
                },
                ...
              ],
              
              "attributes": [
                // an attribute
                {
                  "class_name": str,
                  "source_id": str
                },
                ...
              ],
              
              "transitive_actions": [
                // a transitive action
                {
                  "class_name": str,
                  "source_id": str,
                  "target_id": str
                },
                ...
              ],
              
              "intransitive_actions": [
                // an intransitive action
                {
                  "class_name": str,
                  "source_id": str
                },
                ...
              ]
            }
          ]
        },
        ...
      ]
    }
  },
  ...
]
```