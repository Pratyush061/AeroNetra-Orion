from orion.cli import build_parser, main


def test_list_datasets(capsys):
    assert main(["list-datasets"]) == 0
    out = capsys.readouterr().out
    assert "dsec" in out
    assert "visdrone" in out
    assert "aerial" in out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_demo_smoke(tmp_path):
    cfg = tmp_path / "c.yaml"
    out = tmp_path / "out"
    cfg.write_text(
        "scene:\n  frames: 6\n  width: 160\n  height: 120\n"
        "  num_vehicles: 2\n  num_intruders: 1\n"
        f"output_dir: {out}\nsave_video: false\n"
    )
    assert main(["demo", "--config", str(cfg)]) == 0
    assert (out / "metrics.json").exists()


def test_parser_has_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
