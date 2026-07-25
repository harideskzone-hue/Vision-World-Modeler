# cli.py
import argparse
import sys
import json
import logging
from query.interface import QueryInterface
from shared.enums import NodeStatus

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for noisy_logger in ["httpx", "urllib3", "urllib3.connectionpool", "huggingface_hub", "transformers", "fsspec"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Vision World Modeler CLI Inspector")
    parser.add_argument("--video", type=str, help="Path to video file to process")
    parser.add_argument("--mock", action="store_true", help="Run with mock data (fast test)")
    parser.add_argument("--inspect", type=str, help="Command to run after processing: [entities|graph|history|scene <name>|object <name>|occluded|archived]")
    
    args = parser.parse_args()
    
    if args.mock:
        # Load mock pipeline instead of full vision
        from tests.test_corrections import MockPipeline
        pipeline = MockPipeline()
        pipeline.process(1, "kitchen", [{"name": "mug", "category": "container", "confidence": 0.9}])
        pipeline.process(2, "kitchen", [{"name": "mug", "category": "container", "state": "full", "confidence": 0.95}, {"name": "table", "category": "furniture", "confidence": 0.9}])
        pipeline.process(3, "kitchen", [{"name": "table", "category": "furniture", "confidence": 0.9}]) # Mug occluded
        pipeline.process(35, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}]) # Scene change
        
        graph = pipeline.graph
        registry = pipeline.registry
        current_frame = 35
        print("Processed mock data (35 frames).")
        
    elif args.video:
        from vision_world.orchestrator import VisionOrchestrator
        orchestrator = VisionOrchestrator(args.video)
        graph = orchestrator.run()
        registry = orchestrator.registry
        current_frame = orchestrator.frame_buffer.get_latest().frame_id if orchestrator.frame_buffer.get_latest() else 0
    else:
        print("Please provide --video or --mock")
        sys.exit(1)
        
    query_engine = QueryInterface(graph, registry)
    
    def handle_command(cmd_str: str) -> bool:
        cmd = cmd_str.strip().split()
        if not cmd:
            return True
        action = cmd[0].lower()
        if action in ["exit", "quit", "q"]:
            return False
        elif action == "help":
            print("\nAvailable commands:")
            print("  entities         - List all currently active entities in the world")
            print("  graph            - List all active relationship edges in the world")
            print("  object <name>    - Get full location, state, confidence, and visibility history for an object")
            print("  scene <name>     - List all objects currently in a specific scene/room")
            print("  frame <idx>      - Query exact historical structured state at a specific frame index")
            print("  occluded         - List all currently occluded (temporarily hidden) objects")
            print("  archived         - List entities archived from previous scenes")
            print("  history          - View graph node/edge statistics")
            print("  exit / quit      - Exit interactive mode")
        elif action == "entities":
            print("\nActive Entities:")
            for node in graph.get_all_nodes():
                if node.status == NodeStatus.ACTIVE:
                    print(f"  - {node.id}")
        elif action == "graph":
            print("\nActive Graph Edges:")
            for edge in graph.get_all_active_edges():
                print(f"  {edge.subject} --[{edge.relation.value}]--> {edge.object}")
        elif action == "history":
            print("\nGraph Revision Stats:")
            stats = graph.get_stats()
            print(f"  Total Nodes: {stats.total_nodes}")
            print(f"  Active Edges: {stats.active_edges}")
            print(f"  Superseded Edges: {stats.superseded_edges}")
        elif action == "scene" and len(cmd) > 1:
            scene_name = " ".join(cmd[1:])
            objs = query_engine.get_objects_in_scene(scene_name)
            print(f"\nObjects in '{scene_name}':")
            for o in objs: print(f"  - {o}")
        elif action == "frame" and len(cmd) > 1:
            try:
                f_idx = int(cmd[1])
                state = query_engine.get_state_at_frame(f_idx)
                print(f"\nHistorical World State at Frame {f_idx}:")
                print(f"  Entities: {', '.join(state['entities']) or 'None'}")
                print("  Relationships:")
                for r in state["relationships"]: print(f"    {r}")
            except ValueError:
                print("Invalid frame index. Usage: frame <integer>")
        elif action == "object" and len(cmd) > 1:
            obj_name = " ".join(cmd[1:])
            print("\n" + query_engine.explain(obj_name, current_frame))
        elif action == "occluded":
            objs = query_engine.get_occluded_objects(current_frame)
            print("\nCurrently Occluded Objects:")
            for o in objs: print(f"  - {o}")
        elif action == "archived":
            objs = query_engine.get_archived_entities()
            print("\nArchived Objects:")
            for o in objs: print(f"  - {o}")
        else:
            print(f"Unknown command: '{cmd_str}'. Type 'help' for available commands.")
        return True

    if args.inspect and args.inspect.lower() == "interactive":
        print("\n" + "="*50)
        print("🌍 VISION WORLD MODELER - INTERACTIVE QUERY REPL")
        print("="*50)
        print("Type 'help' to see available query commands, or 'exit' to quit.")
        while True:
            try:
                user_cmd = input("\nWorld> ")
                if not handle_command(user_cmd):
                    print("Exiting interactive world query.")
                    break
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive world query.")
                break
    elif args.inspect:
        handle_command(args.inspect)

if __name__ == "__main__":
    main()
