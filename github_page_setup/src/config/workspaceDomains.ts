/**
 * The 5 workspace domains and canonical workspace surface entity.
 */

export interface WorkspaceDomain {
  id: string;
  name: string;
  description: string;
}

export const WORKSPACE_DOMAINS: WorkspaceDomain[] = [
  {
    id: 'student_workspace',
    name: 'Student Workspace',
    description: 'Study desks, textbooks, stationary, notebooks, and laptops',
  },
  {
    id: 'home_office_workspace',
    name: 'Home Office Workspace',
    description: 'Administrative workstations, printers, documents, and organizers',
  },
  {
    id: 'art_craft_workspace',
    name: 'Art / Craft Workspace',
    description: 'Craft tables, paint tubes, brushes, palettes, and canvases',
  },
  {
    id: 'electronics_maker_workspace',
    name: 'Electronics / Maker Workspace',
    description: 'Prototyping benches, breadboards, microcontrollers, and multimeters',
  },
  {
    id: 'repair_maintenance_workspace',
    name: 'Repair / Maintenance Workspace',
    description: 'Workbenches, screwdrivers, wrenches, pliers, and measuring tools',
  },
];

export const WORKSPACE_SURFACE_ENTITY = {
  instanceId: 'workspace_surface_001',
  categoryId: 'workspace_surface',
  displayName: 'Workspace Surface',
};
