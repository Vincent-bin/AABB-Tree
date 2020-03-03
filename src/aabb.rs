use std::collections::HashMap;
use std::result::Result;

const NULL_NODE: usize = 0xffffffff;

#[derive(Default, Clone)]
pub struct AABB {
  pub surfaceArea: f64,
  pub lowerBound: Vec<f64>,
  pub upperBound: Vec<f64>,
  pub centre: Vec<f64>,
}

impl AABB {
  pub fn new(dimension: usize) -> AABB {
    assert!(dimension >= 2);
    let lowerBound: Vec<f64> = vec![0.0; dimension];
    let upperBound: Vec<f64> = vec![0.0; dimension];
    AABB {
      surfaceArea: 0.0,
      lowerBound: lowerBound,
      upperBound: upperBound,
      centre: vec![],
    }
  }

  pub fn from(lowerBound: &Vec<f64>, upperBound: &Vec<f64>) -> Result<AABB, String> {
    if lowerBound.len() != upperBound.len() {
      return Err("Dimensionality mismatch!".to_owned());
    }

    for i in 0..lowerBound.len() {
      if lowerBound[i] > upperBound[i] {
        return Err("AABB lower bound is greater than the upper bound!".to_owned());
      }
    }
    let surfaceArea = AABB::computeSurfaceArea(lowerBound, upperBound);
    let centre = AABB::computeCentre(lowerBound, upperBound);

    Ok(AABB {
      surfaceArea,
      lowerBound: lowerBound.to_vec(),
      upperBound: upperBound.to_vec(),
      centre,
    })
  }

  pub fn computeSurfaceArea(lowerBound: &Vec<f64>, upperBound: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for d1 in 0..lowerBound.len() {
      let mut product = 1.0;
      for d2 in 0..lowerBound.len() {
        if d1 == d2 {
          continue;
        }
        let dx = upperBound[d2] - lowerBound[d2];
        product *= dx;
      }
      sum += product;
    }
    sum
  }

  pub fn computeCentre(lowerBound: &Vec<f64>, upperBound: &Vec<f64>) -> Vec<f64> {
    let mut position: Vec<f64> = vec![0.0; lowerBound.len()];

    for i in 0..position.len() {
      position[i] = 0.5 * (lowerBound[i] + upperBound[i]);
    }

    position
  }

  pub fn getSurfaceArea(&self) -> f64 {
    self.surfaceArea
  }

  pub fn merge(&mut self, aabb1: &AABB, aabb2: &AABB) {
    assert!(aabb1.lowerBound.len() == aabb2.lowerBound.len());
    assert!(aabb1.upperBound.len() == aabb2.upperBound.len());

    self.lowerBound.resize(aabb1.lowerBound.len(), 0.0);
    self.upperBound.resize(aabb1.lowerBound.len(), 0.0);

    for i in 0..self.lowerBound.len() {
      self.lowerBound[i] = aabb1.lowerBound[i].min(aabb2.lowerBound[i]);
      self.upperBound[i] = aabb1.upperBound[i].max(aabb2.upperBound[i]);
    }

    self.surfaceArea = AABB::computeSurfaceArea(&self.lowerBound, &self.upperBound);
    self.centre = AABB::computeCentre(&self.lowerBound, &self.upperBound);
  }

  pub fn contains(&self, aabb: &AABB) -> bool {
    assert_eq!(self.lowerBound.len(), aabb.lowerBound.len());
    for i in 0..self.lowerBound.len() {
      if aabb.lowerBound[i] < self.lowerBound[i] {
        return false;
      }
      if aabb.upperBound[i] > self.upperBound[i] {
        return false;
      }
    }
    return true;
  }

  pub fn overlaps(&self, aabb: &AABB, touchIsOverlap: bool) -> bool {
    assert_eq!(aabb.lowerBound.len(), self.lowerBound.len());

    let mut rv = true;

    if touchIsOverlap {
      for i in 0..self.lowerBound.len() {
        if aabb.upperBound[i] < self.lowerBound[i] || aabb.lowerBound[i] > self.upperBound[i] {
          rv = false;
          break;
        }
      }
    } else {
      for i in 0..self.lowerBound.len() {
        if aabb.upperBound[i] <= self.lowerBound[i] || aabb.lowerBound[i] >= self.upperBound[i] {
          rv = false;
          break;
        }
      }
    }

    rv
  }

  pub fn setDimension(&mut self, dimension: usize) {
    assert!(dimension >= 2);
    self.lowerBound.resize(dimension, 0.0);
    self.upperBound.resize(dimension, 0.0);
  }
}

#[derive(Default, Clone)]
pub struct Node {
  pub aabb: AABB,
  pub parent: usize,
  pub next: usize,
  pub left: usize,
  pub right: usize,
  pub height: i32,
  pub particle: usize,
}

impl Node {
  pub fn isLeaf(&self) -> bool {
    self.left == NULL_NODE
  }
}

#[derive(Default)]
pub struct Tree {
  root: usize,
  nodes: Vec<Node>,
  nodeCount: usize,
  nodeCapacity: usize,
  freeList: usize,
  dimension: usize,
  isPeriodic: bool,
  skinThickness: f64,
  periodicity: Vec<bool>,
  boxSize: Vec<f64>,
  negMinImage: Vec<f64>,
  posMinImage: Vec<f64>,
  particleMap: HashMap<usize, usize>,
  touchIsOverlap: bool,
}

impl Tree {
  pub fn new(
    dimension: usize,
    skinThickness: f64,
    nParticles: usize,
    touchIsOverlap: bool,
  ) -> Result<Tree, String> {
    if dimension < 2 {
      return Err("Invalid dimensionality!".to_owned());
    }

    let periodicity: Vec<bool> = vec![false; dimension];
    let root = NULL_NODE;
    let nodeCount = 0;
    let nodeCapacity = nParticles;
    let mut nodes: Vec<Node> = vec![Node::default(); nodeCapacity];

    for i in 0..nodeCapacity - 1 {
      nodes[i].next = i + 1;
      nodes[i].height = -1;
    }

    nodes[nodeCapacity - 1].next = NULL_NODE;
    nodes[nodeCapacity - 1].height = -1;
    let freeList = 0;

    Ok(Tree {
      root,
      nodes,
      nodeCount,
      nodeCapacity,
      freeList,
      dimension,
      isPeriodic: false,
      skinThickness,
      periodicity,
      boxSize: vec![],
      negMinImage: vec![],
      posMinImage: vec![],
      particleMap: HashMap::new(),
      touchIsOverlap,
    })
  }

  pub fn from(
    dimension: usize,
    skinThickness: f64,
    periodicity: &Vec<bool>,
    boxSize: &Vec<f64>,
    nParticles: usize,
    touchIsOverlap: bool,
  ) -> Result<Tree, String> {
    if dimension < 2 {
      return Err("Invalid dimensionality!".to_owned());
    }

    if periodicity.len() != dimension || boxSize.len() != dimension {
      return Err("Invalid dimensionality!".to_owned());
    }

    let root = NULL_NODE;
    let nodeCount = 0;
    let nodeCapacity = nParticles;
    let mut nodes: Vec<Node> = vec![Node::default(); nodeCapacity];

    for i in 0..nodeCapacity - 1 {
      nodes[i].next = i + 1;
      nodes[i].height = -1;
    }

    nodes[nodeCapacity - 1].next = NULL_NODE;
    nodes[nodeCapacity - 1].height = -1;
    let freeList = 0;

    let mut isPeriodic = false;

    let mut posMinImage: Vec<f64> = vec![0.0; dimension];
    let mut negMinImage: Vec<f64> = vec![0.0; dimension];

    for i in 0..dimension {
      posMinImage[i] = 0.5 * boxSize[i];
      negMinImage[i] = -0.5 * boxSize[i];
      if (periodicity[i]) {
        isPeriodic = true;
      }
    }

    Ok(Tree {
      root,
      nodes,
      nodeCount,
      nodeCapacity,
      freeList,
      dimension,
      isPeriodic: false,
      skinThickness,
      periodicity: periodicity.to_vec(),
      boxSize: boxSize.to_vec(),
      negMinImage,
      posMinImage,
      particleMap: HashMap::new(),
      touchIsOverlap,
    })
  }

  pub fn setPeriodicity(&mut self, periodicity: &Vec<bool>) {
    self.periodicity = periodicity.to_vec();
  }

  pub fn setBoxSize(&mut self, boxSize: &Vec<f64>) {
    self.boxSize = boxSize.to_vec();
  }

  pub fn allocateNode(&mut self) -> usize {
    if self.freeList == NULL_NODE {
      assert_eq!(self.nodeCount, self.nodeCapacity);
      self.nodeCapacity *= 2;
      self.nodes.resize(self.nodeCapacity, Node::default());

      for i in self.nodeCount..self.nodeCapacity - 1 {
        self.nodes[i].next = i + 1;
        self.nodes[i].height = -1;
      }
      self.nodes[self.nodeCapacity - 1].next = NULL_NODE;
      self.nodes[self.nodeCapacity - 1].height = -1;
      self.freeList = self.nodeCount;
    }

    let node = self.freeList;
    self.freeList = self.nodes[node].next;
    self.nodes[node].parent = NULL_NODE;
    self.nodes[node].left = NULL_NODE;
    self.nodes[node].right = NULL_NODE;
    self.nodes[node].height = 0;
    self.nodes[node].aabb.setDimension(self.dimension);
    self.nodeCount += 1;
    node
  }

  pub fn freeNode(&mut self, node: usize) {
    assert!(node < self.nodeCapacity);
    assert!(0 < self.nodeCount);

    self.nodes[node].next = self.freeList;
    self.nodes[node].height = -1;
    self.freeList = node;
    self.nodeCount -= 1;
  }

  pub fn insertParticle(
    &mut self,
    particle: usize,
    position: &Vec<f64>,
    radius: f64,
  ) -> Result<(), String> {
    if self.particleMap.contains_key(&particle) {
      return Err("Particle already exists in tree!".to_owned());
    }

    if position.len() != self.dimension {
      return Err("Invalid dimensionality!".to_owned());
    }

    let node = self.allocateNode();

    let mut size: Vec<f64> = vec![0.0; self.dimension];

    for i in 0..self.dimension {
      self.nodes[node].aabb.lowerBound[i] = position[i] - radius;
      self.nodes[node].aabb.upperBound[i] = position[i] + radius;
      size[i] = self.nodes[node].aabb.upperBound[i] - self.nodes[node].aabb.lowerBound[i];
    }

    for i in 0..self.dimension {
      self.nodes[node].aabb.lowerBound[i] -= self.skinThickness * size[i];
      self.nodes[node].aabb.upperBound[i] += self.skinThickness * size[i];
    }
    self.nodes[node].aabb.surfaceArea = AABB::computeSurfaceArea(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );
    self.nodes[node].aabb.centre = AABB::computeCentre(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );

    self.nodes[node].height = 0;

    // Insert a new leaf into the tree.
    self.insertLeaf(node);

    // Add the new particle to the map.
    self.particleMap.insert(particle, node);

    // Store the particle index.
    self.nodes[node].particle = particle;

    Ok(())
  }

  pub fn insertParticleWith(
    &mut self,
    particle: usize,
    lowerBound: &Vec<f64>,
    upperBound: &Vec<f64>,
  ) -> Result<(), String> {
    if self.particleMap.contains_key(&particle) {
      return Err("Particle already exists in tree!".to_owned());
    }

    if lowerBound.len() != upperBound.len() {
      return Err("Invalid dimensionality!".to_owned());
    }

    let node = self.allocateNode();

    let mut size: Vec<f64> = vec![0.0; self.dimension];

    for i in 0..self.dimension {
      if lowerBound[i] > upperBound[i] {
        return Err("AABB lower bound is greater than the upper bound!".to_owned());
      }
      self.nodes[node].aabb.lowerBound[i] = lowerBound[i];
      self.nodes[node].aabb.upperBound[i] = upperBound[i];
      size[i] = upperBound[i] - lowerBound[i];
    }

    for i in 0..self.dimension {
      self.nodes[node].aabb.lowerBound[i] -= self.skinThickness * size[i];
      self.nodes[node].aabb.upperBound[i] += self.skinThickness * size[i];
    }
    self.nodes[node].aabb.surfaceArea = AABB::computeSurfaceArea(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );
    self.nodes[node].aabb.centre = AABB::computeCentre(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );

    self.nodes[node].height = 0;

    // Insert a new leaf into the tree.
    self.insertLeaf(node);

    // Add the new particle to the map.
    self.particleMap.insert(particle, node);

    // Store the particle index.
    self.nodes[node].particle = particle;

    Ok(())
  }

  pub fn nParticles(&self) -> usize {
    return self.particleMap.len();
  }

  pub fn removeParticle(&mut self, particle: usize) -> Result<(), String> {
    if !self.particleMap.contains_key(&particle) {
      return Err("Invalid particle index!".to_owned());
    }

    let node = self.particleMap.get(&particle).cloned().unwrap();

    self.particleMap.remove(&particle);

    assert!(node < self.nodeCapacity);
    assert!(self.nodes[node].isLeaf());
    self.removeLeaf(node);
    self.freeNode(node);
    Ok(())
  }

  pub fn removeAll(&mut self) {
    let mut values: Vec<usize> = vec![0, self.particleMap.len()];

    for (key, value) in self.particleMap.iter() {
      values.push(value.clone());
    }

    for node in values {
      assert!(node < self.nodeCapacity);
      assert!(self.nodes[node].isLeaf());

      self.removeLeaf(node);
      self.freeNode(node);
    }
    self.particleMap.clear();
  }

  pub fn updateParticle(
    &mut self,
    particle: usize,
    position: &Vec<f64>,
    radius: f64,
    alwaysReinsert: bool,
  ) -> Result<bool, String> {
    if position.len() != self.dimension {
      return Err(" Dimensionality mismatch!".to_owned());
    }

    let mut lowerBound: Vec<f64> = vec![0.0; self.dimension];
    let mut upperBound: Vec<f64> = vec![0.0; self.dimension];

    for i in 0..self.dimension {
      lowerBound[i] = position[i] - radius;
      upperBound[i] = position[i] + radius;
    }

    self.updateParticleWithBound(particle, &lowerBound, &upperBound, alwaysReinsert)
  }

  pub fn updateParticleWithBound(
    &mut self,
    particle: usize,
    lowerBound: &Vec<f64>,
    upperBound: &Vec<f64>,
    alwaysReinsert: bool,
  ) -> Result<bool, String> {
    if lowerBound.len() != self.dimension && upperBound.len() != self.dimension {
      return Err("Dimensionality mismatch".to_owned());
    }

    let node = self.particleMap.get(&particle).cloned().unwrap();

    assert!(node < self.nodeCapacity);
    assert!(self.nodes[node].isLeaf());

    // AABB size in each dimension.
    let mut size: Vec<f64> = vec![0.0; self.dimension];

    for i in 0..self.dimension {
      if lowerBound[i] > upperBound[i] {
        return Err("AABB lower bound is greater than the upper bound!".to_owned());
      }
      size[i] = upperBound[i] - lowerBound[i];
    }

    let mut aabb = AABB::from(&lowerBound, &upperBound).unwrap();

    if !alwaysReinsert && self.nodes[node].aabb.contains(&aabb) {
      return Ok(false);
    }

    self.removeLeaf(node);

    for i in 0..self.dimension {
      aabb.lowerBound[i] -= self.skinThickness * size[i];
      aabb.upperBound[i] += self.skinThickness * size[i];
    }

    self.nodes[node].aabb = aabb;

    self.nodes[node].aabb.surfaceArea = AABB::computeSurfaceArea(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );
    self.nodes[node].aabb.centre = AABB::computeCentre(
      &self.nodes[node].aabb.lowerBound,
      &self.nodes[node].aabb.upperBound,
    );

    self.insertLeaf(node);

    Ok(true)
  }

  pub fn query(&self, particle: usize) -> Result<Vec<usize>, String> {
    if !self.particleMap.contains_key(&particle) {
      return Err("Invalid particle index!".to_owned());
    }
    let node = self.particleMap.get(&particle).cloned().unwrap();

    self.query_with_aabb(particle, &self.nodes[node].aabb)
  }

  pub fn query_with_aabb(&self, particle: usize, aabb: &AABB) -> Result<Vec<usize>, String> {
    let mut stack: Vec<usize> = vec![];
    stack.reserve(256);
    stack.push(self.root);

    let mut particles: Vec<usize> = vec![];

    while stack.len() > 0 {
      let node = stack.pop().unwrap();

      let mut nodeAABB = self.nodes[node].aabb.clone();

      if node == NULL_NODE {
        continue;
      }

      if self.isPeriodic {
        let mut separation: Vec<f64> = vec![0.0; self.dimension];
        let mut shift: Vec<f64> = vec![0.0; self.dimension];

        for i in 0..self.dimension {
          separation[i] = nodeAABB.centre[i] - aabb.centre[i];
        }

        let isShifted = self.minimumImage(&mut separation, &mut shift);

        if isShifted {
          for i in 0..self.dimension {
            nodeAABB.lowerBound[i] += shift[i];
            nodeAABB.upperBound[i] += shift[i];
          }
        }
      }

      if aabb.overlaps(&nodeAABB, self.touchIsOverlap) {
        if self.nodes[node].isLeaf() {
          if self.nodes[node].particle != particle {
            particles.push(self.nodes[node].particle);
          }
        } else {
          stack.push(self.nodes[node].left);
          stack.push(self.nodes[node].right);
        }
      }
    }
    Ok(particles)
  }

  pub fn query_aabb(&self, aabb: &AABB) -> Result<Vec<usize>, String> {
    if self.particleMap.len() == 0 {
      return Ok(vec![]);
    }

    self.query_with_aabb(std::usize::MAX, &aabb)
  }

  pub fn is_collided(&self, aabb: &AABB) -> bool {
    if self.particleMap.len()==0 {
      return false;
    }

    let particle = std::usize::MAX;

    let mut stack: Vec<usize> = vec![];
    stack.reserve(256);
    stack.push(self.root);

    while stack.len() > 0 {
      let node = stack.pop().unwrap();

      let mut nodeAABB = self.nodes[node].aabb.clone();

      if node == NULL_NODE {
        continue;
      }

      if self.isPeriodic {
        let mut separation: Vec<f64> = vec![0.0; self.dimension];
        let mut shift: Vec<f64> = vec![0.0; self.dimension];

        for i in 0..self.dimension {
          separation[i] = nodeAABB.centre[i] - aabb.centre[i];
        }

        let isShifted = self.minimumImage(&mut separation, &mut shift);

        if isShifted {
          for i in 0..self.dimension {
            nodeAABB.lowerBound[i] += shift[i];
            nodeAABB.upperBound[i] += shift[i];
          }
        }
      }

      if aabb.overlaps(&nodeAABB, self.touchIsOverlap) {
        if self.nodes[node].isLeaf() {
          if self.nodes[node].particle != particle {
            return true;
          }
        } else {
          stack.push(self.nodes[node].left);
          stack.push(self.nodes[node].right);
        }
      }
    }
    false
  }

  // Warning: reference
  pub fn get_aabb(&self, particle: usize) -> &AABB {
    &self.nodes[self.particleMap.get(&particle).cloned().unwrap()].aabb
  }

  pub fn minimumImage(&self, separation: &mut Vec<f64>, shift: &mut Vec<f64>) -> bool {
    let mut isShift = false;

    for i in 0..self.dimension {
      if separation[i] < self.negMinImage[i] {
        separation[i] += if self.periodicity[i] {
          1.0 * self.boxSize[i]
        } else {
          0.0
        };
        shift[i] = if self.periodicity[i] {
          1.0 * self.boxSize[i]
        } else {
          0.0
        };
        isShift = true;
      } else if separation[i] >= self.posMinImage[i] {
        separation[i] -= if self.periodicity[i] {
          1.0 * self.boxSize[i]
        } else {
          0.0
        };
        shift[i] = if self.periodicity[i] {
          -1.0 * self.boxSize[i]
        } else {
          0.0
        };
        isShift = true;
      }
    }

    isShift
  }

  pub fn removeLeaf(&mut self, leaf: usize) {
    if leaf == self.root {
      self.root = NULL_NODE;
      return;
    }

    let parent = self.nodes[leaf].parent;
    let grandParent = self.nodes[parent].parent;
    let mut sibling: usize = 0;

    if self.nodes[parent].left == leaf {
      sibling = self.nodes[parent].right;
    } else {
      sibling = self.nodes[parent].left;
    }

    if grandParent != NULL_NODE {
      if self.nodes[grandParent].left == parent {
        self.nodes[grandParent].left = sibling;
      } else {
        self.nodes[grandParent].right = sibling;
      }

      self.nodes[sibling].parent = grandParent;
      self.freeNode(parent);

      let mut index = grandParent;
      while index != NULL_NODE {
        index = self.balance(index);

        let left = self.nodes[index].left;
        let right = self.nodes[index].right;

        let leftAABB = self.nodes[left].aabb.clone();
        let rightAABB = self.nodes[right].aabb.clone();
        self.nodes[index].aabb.merge(&leftAABB, &rightAABB);
        self.nodes[index].height =
          1 + std::cmp::max(self.nodes[left].height, self.nodes[right].height);

        index = self.nodes[index].parent;
      }
    } else {
      self.root = sibling;
      self.nodes[sibling].parent = NULL_NODE;
      self.freeNode(parent);
    }
  }

  pub fn insertLeaf(&mut self, leaf: usize) {
    if self.root == NULL_NODE {
      self.root = leaf;
      self.nodes[self.root].parent = NULL_NODE;
      return;
    }

    let leafAABB = self.nodes[leaf].aabb.clone();
    let mut index = self.root;

    while !self.nodes[index].isLeaf() {
      let left = self.nodes[index].left;
      let right = self.nodes[index].right;

      let surfaceArea = self.nodes[index].aabb.getSurfaceArea();

      let mut combinedAABB = AABB::default();
      combinedAABB.merge(&self.nodes[index].aabb, &leafAABB);
      let combinedSurfaceArea = combinedAABB.getSurfaceArea();

      let cost = 2.0 * combinedSurfaceArea;

      let inheritanceCost = 2.0 * (combinedSurfaceArea - surfaceArea);

      let mut costLeft = 0.0;
      if self.nodes[left].isLeaf() {
        let mut aabb = AABB::default();
        aabb.merge(&leafAABB, &self.nodes[left].aabb);
        costLeft = aabb.getSurfaceArea() + inheritanceCost;
      } else {
        let mut aabb = AABB::default();
        aabb.merge(&leafAABB, &self.nodes[left].aabb);
        let oldArea = self.nodes[left].aabb.getSurfaceArea();
        let newArea = aabb.getSurfaceArea();
        costLeft = (newArea - oldArea) + inheritanceCost;
      }

      let mut costRight = 0.0;
      if self.nodes[right].isLeaf() {
        let mut aabb = AABB::default();
        aabb.merge(&leafAABB, &self.nodes[right].aabb);
        costRight = aabb.getSurfaceArea() + inheritanceCost;
      } else {
        let mut aabb = AABB::default();
        aabb.merge(&leafAABB, &self.nodes[right].aabb);
        let oldArea = self.nodes[right].aabb.getSurfaceArea();
        let newArea = aabb.getSurfaceArea();
        costRight = (newArea - oldArea) + inheritanceCost;
      }

      if cost < costLeft && cost < costRight {
        break;
      }

      if costLeft < costRight {
        index = left;
      } else {
        index = right;
      }
    }

    let sibling = index;

    let oldParent = self.nodes[sibling].parent;
    let newParent = self.allocateNode();
    self.nodes[newParent].parent = oldParent;
    let siblingAABB = self.nodes[sibling].aabb.clone();
    self.nodes[newParent].aabb.merge(&leafAABB, &siblingAABB);
    self.nodes[newParent].height = self.nodes[sibling].height + 1;

    if oldParent != NULL_NODE {
      if self.nodes[oldParent].left == sibling {
        self.nodes[oldParent].left = newParent;
      } else {
        self.nodes[oldParent].right = newParent;
      }

      self.nodes[newParent].left = sibling;
      self.nodes[newParent].right = leaf;
      self.nodes[sibling].parent = newParent;
      self.nodes[leaf].parent = newParent;
    } else {
      self.nodes[newParent].left = sibling;
      self.nodes[newParent].right = leaf;
      self.nodes[sibling].parent = newParent;
      self.nodes[leaf].parent = newParent;
      self.root = newParent;
    }
    index = self.nodes[leaf].parent;
    while index != NULL_NODE {
      index = self.balance(index);
      let left = self.nodes[index].left;
      let right = self.nodes[index].right;

      assert!(left != NULL_NODE);
      assert!(right != NULL_NODE);

      self.nodes[index].height =
        1 + std::cmp::max(self.nodes[left].height, self.nodes[right].height);

      let leftAABB = self.nodes[left].aabb.clone();
      let rightAABB = self.nodes[right].aabb.clone();

      self.nodes[index].aabb.merge(&leftAABB, &rightAABB);

      index = self.nodes[index].parent;
    }
  }

  pub fn balance(&mut self, node: usize) -> usize {
    assert!(node != NULL_NODE);

    if self.nodes[node].isLeaf() || self.nodes[node].height < 2 {
      return node;
    }

    let left = self.nodes[node].left;
    let right = self.nodes[node].right;

    assert!(left < self.nodeCapacity);
    assert!(right < self.nodeCapacity);

    let currentBalance = self.nodes[right].height - self.nodes[left].height;

    if currentBalance > 1 {
      let rightLeft = self.nodes[right].left;
      let rightRight = self.nodes[right].right;

      assert!(rightLeft < self.nodeCapacity);
      assert!(rightRight < self.nodeCapacity);

      // Swap node and its right-hand child.
      self.nodes[right].left = node;
      self.nodes[right].parent = self.nodes[node].parent;
      self.nodes[node].parent = right;

      if self.nodes[right].parent != NULL_NODE {
        let rightParent = self.nodes[right].parent;
        if self.nodes[self.nodes[right].parent].left == node {
          self.nodes[rightParent].left = right;
        } else {
          assert!(self.nodes[self.nodes[right].parent].right == node);
          self.nodes[rightParent].right = right;
        }
      } else {
        self.root = right;
      }
      // Rotate.
      if self.nodes[rightLeft].height > self.nodes[rightRight].height {
        self.nodes[right].right = rightLeft;
        self.nodes[node].right = rightRight;
        self.nodes[rightRight].parent = node;

        let leftAABB = self.nodes[left].aabb.clone();
        let rightRightAABB = self.nodes[rightRight].aabb.clone();
        self.nodes[node].aabb.merge(&leftAABB, &rightRightAABB);
        let nodeAABB = self.nodes[node].aabb.clone();
        let rightLeftAABB = self.nodes[rightLeft].aabb.clone();
        self.nodes[right].aabb.merge(&nodeAABB, &rightLeftAABB);

        self.nodes[node].height =
          1 + std::cmp::max(self.nodes[left].height, self.nodes[rightRight].height);
        self.nodes[right].height =
          1 + std::cmp::max(self.nodes[node].height, self.nodes[rightLeft].height);
      } else {
        self.nodes[right].right = rightRight;
        self.nodes[node].right = rightLeft;
        self.nodes[rightLeft].parent = node;
        let leftAABB = self.nodes[left].aabb.clone();
        let rightLeftAABB = self.nodes[rightLeft].aabb.clone();
        self.nodes[node].aabb.merge(&leftAABB, &rightLeftAABB);
        let nodeAABB = self.nodes[node].aabb.clone();
        let rightRightAABB = self.nodes[rightRight].aabb.clone();

        self.nodes[right].aabb.merge(&nodeAABB, &rightRightAABB);

        self.nodes[node].height =
          1 + std::cmp::max(self.nodes[left].height, self.nodes[rightLeft].height);
        self.nodes[right].height =
          1 + std::cmp::max(self.nodes[node].height, self.nodes[rightRight].height);
      }

      return right;
    }

    // Rotate left branch up.
    if currentBalance < -1 {
      let leftLeft = self.nodes[left].left;
      let leftRight = self.nodes[left].right;

      assert!(leftLeft < self.nodeCapacity);
      assert!(leftRight < self.nodeCapacity);

      // Swap node and its left-hand child.
      self.nodes[left].left = node;
      self.nodes[left].parent = self.nodes[node].parent;
      self.nodes[node].parent = left;

      // The node's old parent should now point to its left-hand child.
      if self.nodes[left].parent != NULL_NODE {
        let leftParent = self.nodes[left].parent;
        if self.nodes[self.nodes[left].parent].left == node {
          self.nodes[leftParent].left = left;
        } else {
          assert!(self.nodes[self.nodes[left].parent].right == node);
          self.nodes[leftParent].right = left;
        }
      } else {
        self.root = left;
      }
      // Rotate.
      if self.nodes[leftLeft].height > self.nodes[leftRight].height {
        self.nodes[left].right = leftLeft;
        self.nodes[node].left = leftRight;
        self.nodes[leftRight].parent = node;
        let rightAABB = self.nodes[right].aabb.clone();
        let leftRightAABB = self.nodes[leftRight].aabb.clone();
        self.nodes[node].aabb.merge(&rightAABB, &leftRightAABB);
        let nodeAABB = self.nodes[node].aabb.clone();
        let leftLeftAABB = self.nodes[leftLeft].aabb.clone();
        self.nodes[left].aabb.merge(&nodeAABB, &leftLeftAABB);

        self.nodes[node].height =
          1 + std::cmp::max(self.nodes[right].height, self.nodes[leftRight].height);
        self.nodes[left].height =
          1 + std::cmp::max(self.nodes[node].height, self.nodes[leftLeft].height);
      } else {
        self.nodes[left].right = leftRight;
        self.nodes[node].left = leftLeft;
        self.nodes[leftLeft].parent = node;
        let rightAABB = self.nodes[right].aabb.clone();
        let leftLeftAABB = self.nodes[leftLeft].aabb.clone();
        self.nodes[node].aabb.merge(&rightAABB, &leftLeftAABB);
        let nodeAABB = self.nodes[node].aabb.clone();
        let leftRightAABB = self.nodes[leftRight].aabb.clone();

        self.nodes[left].aabb.merge(&nodeAABB, &leftRightAABB);

        self.nodes[node].height =
          1 + std::cmp::max(self.nodes[right].height, self.nodes[leftLeft].height);
        self.nodes[left].height =
          1 + std::cmp::max(self.nodes[node].height, self.nodes[leftRight].height);
      }

      return left;
    }
    node
  }

  pub fn computeHeight(&self) -> usize {
    self.computeHeight_with_node(self.root)
  }

  pub fn computeHeight_with_node(&self, node: usize) -> usize {
    assert!(node < self.nodeCapacity);

    if self.nodes[node].isLeaf() {
      return 0;
    }

    let height1 = self.computeHeight_with_node(self.nodes[node].left);
    let height2 = self.computeHeight_with_node(self.nodes[node].right);

    return 1 + std::cmp::max(height1, height2);
  }

  pub fn getHeight(&self) -> i32 {
    if self.root == NULL_NODE {
      return 0;
    }
    self.nodes[self.root].height
  }

  pub fn getNodeCount(&self) -> usize {
    self.nodeCount
  }

  pub fn computeMaximumBalance(&self) -> usize {
    let mut maxBalance = 0;

    for i in 0..self.nodeCapacity {
      if self.nodes[i].height <= 1 {
        continue;
      }

      assert!(self.nodes[i].isLeaf() == false);

      let balance =
        i32::abs(self.nodes[self.nodes[i].left].height - self.nodes[self.nodes[i].right].height);
    }

    maxBalance
  }

  pub fn computeSurfaceAreaRatio(&self) -> f64 {
    if self.root == NULL_NODE {
      return 0.0;
    }

    let rootArea = AABB::computeSurfaceArea(
      &self.nodes[self.root].aabb.lowerBound,
      &self.nodes[self.root].aabb.upperBound,
    );

    let mut totalArea = 0.0;

    for i in 0..self.nodeCapacity {
      if self.nodes[i].height < 0 {
        continue;
      }

      totalArea += AABB::computeSurfaceArea(
        &self.nodes[i].aabb.lowerBound,
        &self.nodes[i].aabb.upperBound,
      );
    }

    totalArea / rootArea
  }

  pub fn validate(&self) {}

  pub fn rebuild(&mut self) {
    let mut nodeIndices: Vec<usize> = vec![0; self.nodeCount];

    let mut count = 0;

    for i in 0..self.nodeCapacity {
      if self.nodes[i].height < 0 {
        continue;
      }

      if self.nodes[i].isLeaf() {
        self.nodes[i].parent = NULL_NODE;
        nodeIndices[count] = i;
        count += 1;
      } else {
        self.freeNode(i);
      }
    }

    while count > 1 {
      let mut minCost = std::f64::MAX;
      let mut iMin = 0;
      let mut jMin = 0;

      for i in 0..count {
        let aabbi = self.nodes[nodeIndices[i]].aabb.clone();

        for j in i + 1..count {
          let aabbj = self.nodes[nodeIndices[j]].aabb.clone();
          let mut aabb = AABB::default();
          aabb.merge(&aabbi, &aabbj);

          let cost = aabb.getSurfaceArea();
          if cost < minCost {
            iMin = i;
            jMin = j;
            minCost = cost;
          }
        }
      }

      let index1 = nodeIndices[iMin];
      let index2 = nodeIndices[jMin];

      let parent = self.allocateNode();
      self.nodes[parent].left = index1;
      self.nodes[parent].right = index2;
      self.nodes[parent].height =
        1 + std::cmp::max(self.nodes[index1].height, self.nodes[index2].height);
      let index_aabb1 = self.nodes[index1].aabb.clone();
      let index_aabb2 = self.nodes[index2].aabb.clone();
      self.nodes[parent].aabb.merge(&index_aabb1, &index_aabb2);
      self.nodes[parent].parent = NULL_NODE;

      self.nodes[index1].parent = parent;
      self.nodes[index2].parent = parent;

      nodeIndices[jMin] = nodeIndices[count - 1];
      nodeIndices[iMin] = parent;
      count -= 1;
    }

    self.root = nodeIndices[0];
    self.validate()
  }

  pub fn validateStructure(&self, node: usize) {
    if node == NULL_NODE {
      return;
    }

    if node == self.root {
      assert_eq!(self.nodes[node].parent, NULL_NODE);
    }

    let left = self.nodes[node].left;
    let right = self.nodes[node].right;

    if self.nodes[node].isLeaf() {
      assert!(left == NULL_NODE);
      assert!(right == NULL_NODE);
      assert!(self.nodes[node].height == 0);
      return;
    }

    assert!(left < self.nodeCapacity);
    assert!(right < self.nodeCapacity);

    assert!(self.nodes[left].parent == node);
    assert!(self.nodes[right].parent == node);

    self.validateStructure(left);
    self.validateStructure(right);
  }

  pub fn validateMetrics(&self, node: usize) {
    if node == NULL_NODE {
      return;
    }

    let left = self.nodes[node].left;
    let right = self.nodes[node].right;

    if self.nodes[node].isLeaf() {
      assert!(left == NULL_NODE);
      assert!(right == NULL_NODE);
      assert!(self.nodes[node].height == 0);
      return;
    }

    assert!(left < self.nodeCapacity);
    assert!(right < self.nodeCapacity);

    let height1 = self.nodes[left].height;
    let height2 = self.nodes[right].height;

    let height = 1 + std::cmp::max(height1, height2);

    assert!(self.nodes[node].height == height);

    let mut aabb = AABB::default();
    aabb.merge(
      &self.nodes[left].aabb.clone(),
      &self.nodes[right].aabb.clone(),
    );

    for i in 0..self.dimension {
      assert!((aabb.lowerBound[i] - self.nodes[node].aabb.lowerBound[i]).abs() < 1e-5);
      assert!((aabb.upperBound[i] - self.nodes[node].aabb.upperBound[i]).abs() < 1e-5);
    }

    self.validateMetrics(left);
    self.validateMetrics(right);
  }

  pub fn periodicBoundaries(&self, position: &mut Vec<f64>) {
    for i in 0..self.dimension {
      if position[i] < 0.0 {
        position[i] += self.boxSize[i];
      } else {
        if position[i] >= self.boxSize[i] {
          position[i] -= self.boxSize[i];
        }
      }
    }
  }
}

#[test]
fn testTree() {
  let mut tree = Tree::new(3, 0.0, 100, false).unwrap();
}
