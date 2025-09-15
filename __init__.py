import json

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

print("WAS Affine has loaded {} nodes.".format(len(NODE_CLASS_MAPPINGS)))
print("Node Classes Loaded:")
print(json.dumps(list(NODE_CLASS_MAPPINGS.keys()), indent=2))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
