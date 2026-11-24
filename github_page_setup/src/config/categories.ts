/**
 * The 39 project object categories across 5 domains.
 * Fixed ontology: attributes are restricted to color and type.
 */

export interface ObjectCategoryDefinition {
  id: string;
  name: string;
  domain: string;
  aliases: string[];
  defaultAttributes: {
    color: string | null;
    type: string | null;
  };
}

export const WORKSPACE_CATEGORIES: ObjectCategoryDefinition[] = [
  // Common workspace objects (18)
  { id: 'laptop', name: 'Laptop', domain: 'common', aliases: ['notebook computer', 'macbook', 'pc laptop'], defaultAttributes: { color: null, type: null } },
  { id: 'smartphone', name: 'Smartphone', domain: 'common', aliases: ['phone', 'mobile phone', 'cellphone', 'iphone', 'android'], defaultAttributes: { color: null, type: null } },
  { id: 'tablet', name: 'Tablet', domain: 'common', aliases: ['ipad', 'graphic tablet'], defaultAttributes: { color: null, type: null } },
  { id: 'monitor', name: 'Monitor', domain: 'common', aliases: ['display', 'screen', 'computer screen', 'tv'], defaultAttributes: { color: null, type: null } },
  { id: 'keyboard', name: 'Keyboard', domain: 'common', aliases: ['computer keyboard', 'typing keyboard'], defaultAttributes: { color: null, type: null } },
  { id: 'mouse', name: 'Mouse', domain: 'common', aliases: ['computer mouse', 'trackpad'], defaultAttributes: { color: null, type: null } },
  { id: 'notebook', name: 'Notebook', domain: 'common', aliases: ['notepad', 'journal', 'spiral notebook'], defaultAttributes: { color: null, type: null } },
  { id: 'book', name: 'Book', domain: 'common', aliases: ['textbook', 'novel', 'manual'], defaultAttributes: { color: null, type: null } },
  { id: 'loose_paper', name: 'Loose Paper', domain: 'common', aliases: ['paper', 'document', 'sheet of paper', 'notes', 'papers'], defaultAttributes: { color: null, type: null } },
  { id: 'pen', name: 'Pen', domain: 'common', aliases: ['ballpoint pen', 'gel pen', 'ink pen'], defaultAttributes: { color: null, type: null } },
  { id: 'pencil', name: 'Pencil', domain: 'common', aliases: ['lead pencil', 'graphite pencil'], defaultAttributes: { color: null, type: null } },
  { id: 'marker', name: 'Marker', domain: 'common', aliases: ['highlighter', 'felt pen', 'sharpie'], defaultAttributes: { color: null, type: null } },
  { id: 'calculator', name: 'Calculator', domain: 'common', aliases: ['scientific calculator'], defaultAttributes: { color: null, type: null } },
  { id: 'device_cable', name: 'Device Cable', domain: 'common', aliases: ['cable', 'usb cable', 'cord', 'wire', 'hdmi cable', 'lightning cable'], defaultAttributes: { color: null, type: null } },
  { id: 'charger', name: 'Charger', domain: 'common', aliases: ['power adapter', 'phone charger', 'laptop charger', 'wall charger', 'charging brick'], defaultAttributes: { color: null, type: null } },
  { id: 'water_bottle', name: 'Water Bottle', domain: 'common', aliases: ['bottle', 'flask', 'tumbler', 'drink bottle'], defaultAttributes: { color: null, type: null } },
  { id: 'scissors', name: 'Scissors', domain: 'common', aliases: ['shears', 'pair of scissors'], defaultAttributes: { color: null, type: null } },
  { id: 'headphones', name: 'Headphones', domain: 'common', aliases: ['earphones', 'headset', 'airpods', 'earbuds'], defaultAttributes: { color: null, type: null } },

  // Student workspace (3)
  { id: 'ruler', name: 'Ruler', domain: 'student', aliases: ['scale', 'measuring ruler'], defaultAttributes: { color: null, type: null } },
  { id: 'eraser', name: 'Eraser', domain: 'student', aliases: ['rubber'], defaultAttributes: { color: null, type: null } },
  { id: 'pencil_sharpener', name: 'Pencil Sharpener', domain: 'student', aliases: ['sharpener'], defaultAttributes: { color: null, type: null } },

  // Home/office/administrative (3)
  { id: 'printer', name: 'Printer', domain: 'home_office', aliases: ['scanner', 'copier'], defaultAttributes: { color: null, type: null } },
  { id: 'document_folder', name: 'Document Folder', domain: 'home_office', aliases: ['folder', 'file folder', 'binder'], defaultAttributes: { color: null, type: null } },
  { id: 'stapler', name: 'Stapler', domain: 'home_office', aliases: ['staple gun'], defaultAttributes: { color: null, type: null } },

  // Art/craft (5)
  { id: 'paintbrush', name: 'Paintbrush', domain: 'art_craft', aliases: ['brush', 'art brush'], defaultAttributes: { color: null, type: null } },
  { id: 'paint_tube', name: 'Paint Tube', domain: 'art_craft', aliases: ['acrylic tube', 'oil paint'], defaultAttributes: { color: null, type: null } },
  { id: 'paint_palette', name: 'Paint Palette', domain: 'art_craft', aliases: ['palette', 'color palette'], defaultAttributes: { color: null, type: null } },
  { id: 'canvas', name: 'Canvas', domain: 'art_craft', aliases: ['painting canvas', 'art board'], defaultAttributes: { color: null, type: null } },
  { id: 'glue_bottle', name: 'Glue Bottle', domain: 'art_craft', aliases: ['glue', 'adhesive bottle'], defaultAttributes: { color: null, type: null } },

  // Electronics/maker (5)
  { id: 'arduino_board', name: 'Arduino Board', domain: 'electronics_maker', aliases: ['arduino', 'microcontroller', 'esp32', 'raspberry pi'], defaultAttributes: { color: null, type: null } },
  { id: 'breadboard', name: 'Breadboard', domain: 'electronics_maker', aliases: ['prototyping board', 'solderless breadboard'], defaultAttributes: { color: null, type: null } },
  { id: 'sensor_module', name: 'Sensor Module', domain: 'electronics_maker', aliases: ['sensor', 'breakout board'], defaultAttributes: { color: null, type: null } },
  { id: 'multimeter', name: 'Multimeter', domain: 'electronics_maker', aliases: ['digital multimeter', 'voltmeter', 'dmm'], defaultAttributes: { color: null, type: null } },
  { id: 'jumper_wire', name: 'Jumper Wire', domain: 'electronics_maker', aliases: ['jumper cables', 'breadboard wire', 'dupont wire'], defaultAttributes: { color: null, type: null } },

  // Repair/maintenance (5)
  { id: 'screwdriver', name: 'Screwdriver', domain: 'repair_maintenance', aliases: ['phillips screwdriver', 'flathead screwdriver'], defaultAttributes: { color: null, type: null } },
  { id: 'pliers', name: 'Pliers', domain: 'repair_maintenance', aliases: ['needle-nose pliers', 'cutting pliers'], defaultAttributes: { color: null, type: null } },
  { id: 'wrench', name: 'Wrench', domain: 'repair_maintenance', aliases: ['spanner', 'adjustable wrench'], defaultAttributes: { color: null, type: null } },
  { id: 'hammer', name: 'Hammer', domain: 'repair_maintenance', aliases: ['mallet', 'claw hammer'], defaultAttributes: { color: null, type: null } },
  { id: 'measuring_tape', name: 'Measuring Tape', domain: 'repair_maintenance', aliases: ['tape measure'], defaultAttributes: { color: null, type: null } },
];

export const CATEGORY_BY_ID = new Map<string, ObjectCategoryDefinition>(
  WORKSPACE_CATEGORIES.map((c) => [c.id, c])
);
