--[[ Copyright 2020 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

-- Tests for the `avatar_library` module.

local meltingpot = 'meltingpot.lua.modules.'
local base_simulation = require(meltingpot .. 'base_simulation')
local avatar_library = require(meltingpot .. 'avatar_library')
local game_object = require(meltingpot .. 'game_object')

local grid_world = require 'system.grid_world'
local random = require 'system.random'
local tile_set = require 'common.tile_set'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'

local SPRITE_SHAPE = [[
@@@@@@@@@@@@@@@@
@ABABAABBBABBBB@
@ABBBAABBBABABB@
@BAABABBBABAAAA@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@xxxxxxxxxxxxxx@
@@@@@@@@@@@@@@@@
]]

local PALETTE = {
    ['A'] = {20, 40, 60, 255},
    ['B'] = {5, 5, 5, 255},
    ['x'] = {0, 0, 0, 0},
    ['@'] = {23, 11, 19, 255},
}

local function getEmptyWorldConfig()
  local worldConfig = {
    outOfBoundsSprite = 'OutOfBounds',
    outOfViewSprite = 'OutOfView',
    updateOrder = {},
    renderOrder = {'logic', 'lowerPhysical', 'upperPhysical'},
    customSprites = {},
    hits = {},
    states = {}
  }
  return worldConfig
end


local tests = {}

function tests.zapper()
  local component = avatar_library.Zapper{
      cooldownTime = 86,
      beamLength = 13,
      beamRadius = 9,
      framesTillRespawn = 92,
      penaltyForBeingZapped = -5,
      rewardForZapping = 5
  }

  local worldConfig = getEmptyWorldConfig()
  local world = grid_world.World(worldConfig)
  local tileSet = tile_set.TileSet(world, {width = 5, height = 1})

  component:addHits(worldConfig)

  local expectedHits = {
      ['zapHit'] = {
        layer = 'beamZap',
        sprite = 'BeamZap',
      }
  }
  log.info(helpers.tostring(worldConfig.hits))
  asserts.tablesEQ(worldConfig.hits, expectedHits)

  local expectedRenderOrder = {'logic', 'lowerPhysical', 'upperPhysical',
                               'beamZap'}
  asserts.tablesEQ(worldConfig.renderOrder, expectedRenderOrder)
end

return test_runner.run(tests)
