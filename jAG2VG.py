import pickle
from PIL import Image
import json
import numpy as np
import os
import argparse

def format_ag_to_vg(args):
    source_dir = args.source_dir # default="ActionGenome/dataset/ag/"
    dest_dir = args.dest_dir # default="AGinVGformat/"
    
    if ~os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if ~os.path.exists(dest_dir+'VG_100K'):
        os.mkdir(dest_dir+'VG_100K')
    file1 = open(source_dir+'annotations/object_bbox_and_relationship.pkl', 'rb')
    file2 = open(source_dir+'annotations/person_bbox.pkl', 'rb')
    obj_rel_file = pickle.load(file1)
    person_file = pickle.load(file2)
    obj_list = ['person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
                'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries', 'laptop',
                'light', 'medicine', 'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator',
                'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window']
    rel_list = ['looking_at', 'not_looking_at', 'unsure', 'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of',
                'in', 'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on',
                'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting',
                'wearing', 'wiping', 'writing_on']
    # rel types: 1 = attention, 2 = spatial, 3 = contacting relation
    #            0 = unsure, notcontacting or otherrelationship
    rel_list_type = [1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3]
    rel_skip_list = ['unsure', 'not_contacting', 'other_relationship']
    image_id = 0

    # Output JSON objects / dicts
    image_data = []
    objects = []
    attributes = []
    relationships = []  # spatial, attention and contacting relationships
    scene_graphs = []
    lists_object_relation = {
        "objects_list": obj_list,
        "relationships_list": rel_list,
        "relationships_list_types": rel_list_type
    }

    # for attribute_synsets.json
    attribute_synsets = {"nothing": "nothing.n.01"}

    for key in obj_rel_file.keys():
        obj_rel_dict_list = obj_rel_file[key]
        person_dict = person_file[key]
        subj_id = obj_list.index('person')  # Person = Subject
        subj_bbox = person_dict['bbox']  # XYXY box format (top-left corner and bottom-right corner)
        if np.shape(subj_bbox)[0]>0:
            subj_bbox = subj_bbox.astype(int).tolist()
            subj_bbox[0][2] = subj_bbox[0][2] - subj_bbox[0][0]  # to XYWH box format
            subj_bbox[0][3] = subj_bbox[0][3] - subj_bbox[0][1]
            if os.path.exists(source_dir + 'frames/' + key):
                I = Image.open(source_dir + 'frames/' + key)
                [w, h] = I.size
                key_split = key.split('/')
                filename = key_split[0] + '_' + key_split[1]
                I.save(dest_dir + 'VG_100K/' + filename)
                os.remove(source_dir + 'frames/' + key)
                image_id = image_id + 1

                # For image_data.json
                image_data.append({
                    "image_id": image_id,
                    "filename": filename,
                    "url": None,
                    "width": w,
                    "height": h,
                    "coco_id": None,
                    "flicker_id": None
                })

                for i in range(len(obj_rel_dict_list)):
                    if obj_rel_dict_list[i]['bbox'] is not None:
                        obj_rel_dict = obj_rel_dict_list[i]
                        obj_name = obj_rel_dict['class']
                        obj_id = obj_list.index(obj_name)
                        obj_bbox = np.array(obj_rel_dict['bbox'])  # XYWH box format (top-left corner, width, height)
                        obj_bbox = obj_bbox.astype(int).tolist()
                        sp_rel = obj_rel_dict['spatial_relationship']
                        at_rel = obj_rel_dict['attention_relationship']
                        cont_rel = obj_rel_dict['contacting_relationship'] 
                        visible = obj_rel_dict['visible']
                        # if relation is visible, format data in dicts and to vars for JSON files
                        if visible and (any(item not in rel_skip_list for item in sp_rel) or any(item not in rel_skip_list for item in at_rel)):

                            # for objects.json
                            objs = []
                            objs.append({
                                "object_id": obj_id,
                                "x": obj_bbox[0],
                                "y": obj_bbox[1],
                                "w": obj_bbox[2],
                                "h": obj_bbox[3],
                                "name": obj_name,
                                "synsets": [obj_name + ".n.01"]
                            })
                            objs.append({
                                "object_id": subj_id,
                                "x": subj_bbox[0][0],
                                "y": subj_bbox[0][1],
                                "w": subj_bbox[0][2],
                                "h": subj_bbox[0][3],
                                "name": "person",
                                "synsets": ["person.n.01"]
                            })
                            objects.append({
                                "image_id": image_id,
                                "filename": filename,
                                "objects": objs
                            })

                            # for attributes.json
                            attrs = []
                            attrs.append({
                                "object_id": obj_id,
                                "x": obj_bbox[0],
                                "y": obj_bbox[1],
                                "w": obj_bbox[2],
                                "h": obj_bbox[3],
                                "name": obj_name,
                                "synsets": [obj_name + ".n.01"],
                                "attributes": "nothing"
                            })
                            attrs.append({
                                "object_id": subj_id,
                                "x": subj_bbox[0][0],
                                "y": subj_bbox[0][1],
                                "w": subj_bbox[0][2],
                                "h": subj_bbox[0][3],
                                "name": "person",
                                "synsets": ["person.n.01"],
                                "attributes": "nothing"
                            })
                            attributes.append({
                                "image_id": image_id,
                                "filename": filename,
                                "attributes": attrs
                            })

                            # for relationships.json
                            rels = []
                            for rel in sp_rel:
                                if rel not in rel_skip_list:
                                    rels.append({
                                        "relationship_id": rel_list.index(rel),
                                        "predicate": rel,
                                        "synsets": [rel + ".v.01"],
                                        "subject": {
                                            "object_id": subj_id,
                                            "x": subj_bbox[0][0],
                                            "y": subj_bbox[0][1],
                                            "w": subj_bbox[0][2],
                                            "h": subj_bbox[0][3],
                                            "name": "person",
                                            "synsets": ["person.n.01"]
                                        },
                                        "object": {
                                            "object_id": obj_id,
                                            "x": obj_bbox[0],
                                            "y": obj_bbox[1],
                                            "w": obj_bbox[2],
                                            "h": obj_bbox[3],
                                            "name": obj_name,
                                            "synsets": [obj_name + ".n.01"]
                                        }
                                    })
                            for rel in at_rel:
                                if rel not in rel_skip_list:
                                    rels.append({
                                        "relationship_id": rel_list.index(rel),
                                        "predicate": rel,
                                        "synsets": [rel + ".v.01"],
                                        "subject": {
                                            "object_id": subj_id,
                                            "x": subj_bbox[0][0],
                                            "y": subj_bbox[0][1],
                                            "w": subj_bbox[0][2],
                                            "h": subj_bbox[0][3],
                                            "name": "person",
                                            "synsets": ["person.n.01"]
                                        },
                                        "object": {
                                            "object_id": obj_id,
                                            "x": obj_bbox[0],
                                            "y": obj_bbox[1],
                                            "w": obj_bbox[2],
                                            "h": obj_bbox[3],
                                            "name": obj_name,
                                            "synsets": [obj_name + ".n.01"]
                                        }
                                    })

                            for rel in cont_rel:
                                if rel not in rel_skip_list:
                                    rels.append({
                                        "relationship_id": rel_list.index(rel),
                                        "predicate": rel,
                                        "synsets": [rel + ".v.01"],
                                        "subject": {
                                            "object_id": subj_id,
                                            "x": subj_bbox[0][0],
                                            "y": subj_bbox[0][1],
                                            "w": subj_bbox[0][2],
                                            "h": subj_bbox[0][3],
                                            "name": "person",
                                            "synsets": ["person.n.01"]
                                        },
                                        "object": {
                                            "object_id": obj_id,
                                            "x": obj_bbox[0],
                                            "y": obj_bbox[1],
                                            "w": obj_bbox[2],
                                            "h": obj_bbox[3],
                                            "name": obj_name,
                                            "synsets": [obj_name + ".n.01"]
                                        }
                                    })
                            relationships.append({
                                "image_id": image_id,
                                "filename": filename,
                                "relationships": rels
                            })

                            # for scene_graphs.json
                            sg_rels = []
                            for rel in sp_rel:
                                if rel not in rel_skip_list:
                                    sg_rels.append({
                                        "relationship_id": rel_list.index(rel),
                                        "predicate": rel,
                                        "synsets": [rel + ".v.01"],
                                        "subject_id": subj_id,
                                        "object_id": obj_id
                                    })
                            for rel in at_rel:
                                if rel not in rel_skip_list:
                                    sg_rels.append({
                                        "relationship_id": rel_list.index(rel),
                                        "predicate": rel,
                                        "synsets": [rel + ".v.01"],
                                        "subject_id": subj_id,
                                        "object_id": obj_id
                                    })
                            scene_graphs.append({
                                "image_id": image_id,
                                "filename": filename,
                                "objects": objs,
                                "relationships": sg_rels
                            })
                print(key + ' processed... ' + str(image_id))
            else:
                print(key + ' not found')

    # save data formatted for JSONs to JSON files
    with open(dest_dir+'image_data.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(image_data))
        else:
            print('image_data.json already exists, not updated.')
    with open(dest_dir+'objects.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(objects))
        else:
            print('objects.json already exists, not updated.')
    with open(dest_dir+'attributes.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(attributes))
        else:
            print('attributes.json already exists, not updated.')
    with open(dest_dir+'relationships.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(relationships))
        else:
            print('relationships.json already exists, not updated.')
    with open(dest_dir+'scene_graphs.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(scene_graphs))
        else:
            print('scene_graphs.json already exists, not updated.')
    with open(dest_dir+'lists_object_relation.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(lists_object_relation))
        else:
            print('lists_object_relation.json already exists, not updated.')
    with open(dest_dir+'attribute_synsets.json', 'w+') as f:
        if len(f.read()) == 0:
            f.write(json.dumps(attribute_synsets))
        else:
            print('attribute_synsets.json already exists, not updated.')


    file1.close()
    file2.close()

    # Code to write to JSON
    # with open('logs.json', 'r+') as f:
    #    if len(f.read()) == 0:
    #        f.write(json.dumps(dictionary))
    #    else:
    #        f.write(',\n' + json.dumps(dictionary))

    # Code to read from JSON
    # with open('logs.json') as f:
    #    g = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format AG dataset to VG format")
    parser.add_argument("--source_dir", default="ActionGenome/dataset/ag/",
                        help="Folder containing AG videos, frames and annotations folders.")
    parser.add_argument("--dest_dir", default="AGinVGformat/",
                        help="Folder where VG images and JSON files will be saved.")
    args = parser.parse_args()
    format_ag_to_vg(args)
