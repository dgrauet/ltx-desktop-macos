#!/usr/bin/env ruby
# Usage: ruby app/scripts/add_swift_file.rb LTXDesktop/Services/Foo.swift
# Registers a Swift file in the LTXDesktop target's Sources build phase (idempotent).
require 'xcodeproj'

rel = ARGV[0] or abort "usage: add_swift_file.rb <path-relative-to-app>"
proj_path = File.expand_path('../LTXDesktop.xcodeproj', __dir__)
proj = Xcodeproj::Project.open(proj_path)
target = proj.targets.find { |t| t.name == 'LTXDesktop' } or abort "LTXDesktop target not found"

base = File.basename(rel)
if target.source_build_phase.files.any? { |bf| bf.file_ref&.path&.to_s&.end_with?(base) }
  puts "already present: #{rel}"
else
  ref = proj.main_group.new_file(rel)
  target.add_file_references([ref])
  proj.save
  puts "added: #{rel}"
end
