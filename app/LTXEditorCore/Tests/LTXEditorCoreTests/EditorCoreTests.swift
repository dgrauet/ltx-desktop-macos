import XCTest
@testable import LTXEditorCore

final class TimelineOperationTests: XCTestCase {
    private func project(videos: [Double] = [4, 2], music: Double? = nil) -> (Project, [UUID]) {
        var p = Project(name: "Test")
        var ids: [UUID] = []
        for (i, d) in videos.enumerated() {
            ids.append(p.addAsset(Asset(path: "/v\(i).mp4", kind: .video, name: "v\(i)", duration: d, hasAudio: true)).id)
        }
        if let music {
            ids.append(p.addAsset(Asset(path: "/m.wav", kind: .audio, name: "music", duration: music, hasAudio: true)).id)
        }
        return (p, ids)
    }

    func testAddAssetDedupesByPath() {
        var p = Project(name: "x")
        let a = p.addAsset(Asset(path: "/a.mp4", kind: .video, name: "a", duration: 1, hasAudio: false))
        let b = p.addAsset(Asset(path: "/a.mp4", kind: .video, name: "again", duration: 1, hasAudio: false))
        XCTAssertEqual(a.id, b.id)
        XCTAssertEqual(p.assets.count, 1)
    }

    func testAppendAndDuration() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        try p.insertVideoClip(assetId: ids[1])
        XCTAssertEqual(p.timeline.duration, 6, accuracy: 1e-9)
        XCTAssertEqual(p.timeline.videoClipStarts, [0, 4])
    }

    func testSplitAndRippleDelete() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        try p.insertVideoClip(assetId: ids[1])
        let second = try p.splitVideoClip(atTime: 1.5)
        XCTAssertEqual(p.timeline.videoClips.count, 3)
        XCTAssertEqual(p.timeline.videoClips[0].outPoint, 1.5, accuracy: 1e-9)
        XCTAssertEqual(p.timeline.videoClips[1].inPoint, 1.5, accuracy: 1e-9)
        XCTAssertEqual(p.timeline.duration, 6, accuracy: 1e-9)

        try p.deleteVideoClip(second)
        XCTAssertEqual(p.timeline.duration, 3.5, accuracy: 1e-9)
        XCTAssertEqual(p.timeline.videoClipStarts, [0, 1.5])
    }

    func testSplitAtBoundaryIsRejected() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        XCTAssertThrowsError(try p.splitVideoClip(atTime: 0)) { XCTAssertEqual($0 as? EditorError, .nothingToSplit) }
        XCTAssertThrowsError(try p.splitVideoClip(atTime: 10)) { XCTAssertEqual($0 as? EditorError, .nothingToSplit) }
    }

    func testInsertAtTimeSplitsTheClipUnderIt() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        let inserted = try p.insertVideoClip(assetId: ids[1], atTime: 1)
        XCTAssertEqual(p.timeline.videoClips.map(\.duration), [1, 2, 3])
        XCTAssertEqual(p.timeline.videoClips[1].id, inserted)
    }

    func testInsertAtBoundaryDoesNotSplit() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        try p.insertVideoClip(assetId: ids[1], atTime: 0)
        XCTAssertEqual(p.timeline.videoClips.map(\.duration), [2, 4])
    }

    func testMoveClip() throws {
        var (p, ids) = project()
        let a = try p.insertVideoClip(assetId: ids[0])
        let b = try p.insertVideoClip(assetId: ids[1])
        try p.moveVideoClip(b, to: 0)
        XCTAssertEqual(p.timeline.videoClips.map(\.id), [b, a])
        try p.moveVideoClip(b, to: 99)
        XCTAssertEqual(p.timeline.videoClips.map(\.id), [a, b])
    }

    func testTrimClampsAndRejectsSubFrame() throws {
        var (p, ids) = project()
        let a = try p.insertVideoClip(assetId: ids[0])
        try p.trimVideoClip(a, inPoint: -1, outPoint: 99)
        XCTAssertEqual(p.timeline.videoClips[0].inPoint, 0)
        XCTAssertEqual(p.timeline.videoClips[0].outPoint, 4)
        let before = p
        XCTAssertThrowsError(try p.trimVideoClip(a, inPoint: 2, outPoint: 2.01))
        XCTAssertEqual(p, before, "a failed edit must leave the project unchanged")
    }

    func testTakesSwitchAndClampRange() throws {
        var (p, ids) = project(videos: [4, 1])
        let clip = try p.insertVideoClip(assetId: ids[0])
        try p.trimVideoClip(clip, inPoint: 0.5, outPoint: 3.5)
        let retake = try p.addTake(toClip: clip, assetId: ids[1], label: "Retake 1")
        let c = p.timeline.videoClips[0]
        XCTAssertEqual(c.activeTakeId, retake)
        XCTAssertEqual(c.takes.count, 2)
        XCTAssertEqual(c.outPoint, 1, accuracy: 1e-9, "range clamped to the shorter take")
        XCTAssertEqual(c.inPoint, 0.5, accuracy: 1e-9)

        try p.activateTake(c.takes[0].id, inClip: clip)
        XCTAssertEqual(p.timeline.videoClips[0].activeTake.label, "Original")
    }

    func testAssetInUseCannotBeRemoved() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        XCTAssertThrowsError(try p.removeAsset(ids[0])) { XCTAssertEqual($0 as? EditorError, .assetInUse) }
        try p.removeAsset(ids[1])
        XCTAssertEqual(p.assets.count, 1)
    }

    func testWrongKindRejected() {
        var (p, ids) = project(music: 10)
        XCTAssertThrowsError(try p.insertVideoClip(assetId: ids[2])) { XCTAssertEqual($0 as? EditorError, .wrongAssetKind) }
    }

    func testMusicExtendsDurationAndClamps() throws {
        var (p, ids) = project(music: 10)
        try p.insertVideoClip(assetId: ids[0])
        let m = try p.addAudioClip(assetId: ids[2], start: 1)
        XCTAssertEqual(p.timeline.duration, 11, accuracy: 1e-9)
        try p.moveAudioClip(m, start: -5)
        XCTAssertEqual(p.timeline.audioClips[0].start, 0)
        try p.setAudioClipVolume(m, 3)
        XCTAssertEqual(p.timeline.audioClips[0].volume, 1)
    }

    func testClipAtTime() throws {
        var (p, ids) = project()
        try p.insertVideoClip(assetId: ids[0])
        try p.insertVideoClip(assetId: ids[1])
        XCTAssertEqual(p.timeline.videoClip(at: 5)?.index, 1)
        XCTAssertEqual(p.timeline.videoClip(at: 5)?.offset ?? -1, 1, accuracy: 1e-9)
        XCTAssertNil(p.timeline.videoClip(at: 6))
    }
}

final class UndoStackTests: XCTestCase {
    func testUndoRedo() {
        var stack = UndoStack<Int>(limit: 3)
        var value = 0
        for next in 1...5 {
            stack.record(value)
            value = next
        }
        XCTAssertEqual(stack.past, [2, 3, 4], "oldest snapshots dropped past the limit")
        value = stack.undo(from: value)!
        XCTAssertEqual(value, 4)
        value = stack.redo(from: value)!
        XCTAssertEqual(value, 5)
        stack.record(value)
        XCTAssertFalse(stack.canRedo, "a new edit clears redo")
    }
}

final class ProjectStoreTests: XCTestCase {
    func testRoundTripListDelete() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ProjectStore(root: root)

        var p = Project(name: "Trip")
        let a = p.addAsset(Asset(path: "/a b.mp4", kind: .video, name: "a", duration: 2, hasAudio: true))
        try p.insertVideoClip(assetId: a.id)
        try store.save(p)

        let loaded = try store.load(p.id)
        XCTAssertEqual(loaded.timeline, p.timeline)
        XCTAssertEqual(loaded.assets, p.assets)
        XCTAssertEqual(store.list().map(\.name), ["Trip"])
        XCTAssertEqual(store.list().first?.clipCount, 1)

        try FileManager.default.createDirectory(at: root.appendingPathComponent("junk"), withIntermediateDirectories: true)
        XCTAssertEqual(store.list().count, 1, "non-project directories are ignored")

        try store.delete(p.id)
        XCTAssertTrue(store.list().isEmpty)
    }
}

final class FCPXMLWriterTests: XCTestCase {
    func testDocumentStructure() throws {
        var p = Project(name: "Edit & <cut>")
        let v = p.addAsset(Asset(path: "/clips/fox run.mp4", kind: .video, name: "fox", duration: 4, hasAudio: true, width: 768, height: 512))
        let m = p.addAsset(Asset(path: "/m.wav", kind: .audio, name: "music", duration: 10, hasAudio: true))
        let clip = try p.insertVideoClip(assetId: v.id)
        try p.trimVideoClip(clip, inPoint: 1, outPoint: 3)
        let music = try p.addAudioClip(assetId: m.id, start: 0.5)
        try p.setAudioClipVolume(music, 0.5)

        let xml = FCPXMLWriter.document(for: p)
        let doc = try XMLDocument(xmlString: xml)  // well-formed
        XCTAssertEqual(try doc.nodes(forXPath: "//spine/asset-clip").count, 1)
        XCTAssertEqual(try doc.nodes(forXPath: "//spine/gap").count, 1, "music runs past the video: trailing gap")
        let spineClip = try XCTUnwrap(try doc.nodes(forXPath: "//spine/asset-clip").first as? XMLElement)
        XCTAssertEqual(spineClip.attribute(forName: "start")?.stringValue, "24/24s")
        XCTAssertEqual(spineClip.attribute(forName: "duration")?.stringValue, "48/24s")
        let connected = try XCTUnwrap(try doc.nodes(forXPath: "//spine/asset-clip/asset-clip").first as? XMLElement)
        XCTAssertEqual(connected.attribute(forName: "lane")?.stringValue, "-1")
        XCTAssertEqual(connected.attribute(forName: "offset")?.stringValue, "36/24s", "parent source start 1 s + 0.5 s")
        XCTAssertTrue(xml.contains("-6.0dB"))
        XCTAssertTrue(xml.contains("file:///clips/fox%20run.mp4"))
        XCTAssertTrue(xml.contains("Edit &amp; &lt;cut&gt;"))
        XCTAssertTrue(xml.contains("width=\"768\""))
    }

    func testMutedVideoTrack() throws {
        var p = Project(name: "m")
        let v = p.addAsset(Asset(path: "/v.mp4", kind: .video, name: "v", duration: 1, hasAudio: true))
        try p.insertVideoClip(assetId: v.id)
        p.timeline.videoTrackMuted = true
        XCTAssertTrue(FCPXMLWriter.document(for: p).contains("-96dB"))
    }
}
